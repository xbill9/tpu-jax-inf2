"""Unit tests for the AWS Inf2 deployment scaffold (no AWS calls)."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "deployments" / "aws-inf2"


def load(name, filename):
    path = DEPLOY_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


deploy = load("aws_inf2_deploy", "deploy.py")
# Safe to import without JAX installed: the jax import lives inside
# verify_neuron(), not at module scope.
entrypoint = load("aws_inf2_entrypoint", "neuron_entrypoint.py")


def config(**overrides):
    values = dict(
        region="us-east-1",
        project="test-gemma",
        source_uri="s3://bucket/source.tar.gz",
        subnet_id="subnet-1",
        security_group_id="sg-1",
        instance_profile_name="profile",
        instance_type="inf2.xlarge",
        market_type="on-demand",
        model_id="google/gemma-4-E2B-it-qat-w4a16-ct",
        hf_secret_id="hf-token",
        max_model_len=4096,
        volume_gib=200,
        cache_volume_gib=200,
        swap_gib=32,
        neuron_cc_flags="--model-type=transformer",
        ami_id="ami-test",
    )
    values.update(overrides)
    return deploy.Config(**values)


class FakeEc2:
    def __init__(self, instances=None, volumes=None, az="us-east-1a"):
        self.instances = instances or []
        self.volumes = volumes or []
        self.az = az
        self.run_request = None
        self.terminated = None
        self.created_volume = None
        self.attached = None
        self.deleted_volumes = []

    def describe_instances(self, **_kwargs):
        return {"Reservations": [{"Instances": self.instances}]}

    def describe_subnets(self, **_kwargs):
        return {"Subnets": [{"AvailabilityZone": self.az}]}

    def describe_volumes(self, **kwargs):
        if "VolumeIds" in kwargs:
            wanted = set(kwargs["VolumeIds"])
            return {"Volumes": [v for v in self.volumes if v["VolumeId"] in wanted]}
        return {"Volumes": list(self.volumes)}

    def create_volume(self, **kwargs):
        self.created_volume = kwargs
        return {"VolumeId": "vol-created"}

    def attach_volume(self, **kwargs):
        self.attached = kwargs
        return {}

    def delete_volume(self, **kwargs):
        self.deleted_volumes.append(kwargs["VolumeId"])
        return {}

    def get_waiter(self, _name):
        return mock.Mock()

    def run_instances(self, **kwargs):
        self.run_request = kwargs
        return {"Instances": [{"InstanceId": "i-new"}]}

    def terminate_instances(self, **kwargs):
        self.terminated = kwargs
        return {"TerminatingInstances": []}


def volume(volume_id="vol-cache", az="us-east-1a", state="available", created=1):
    return {"VolumeId": volume_id, "AvailabilityZone": az, "State": state,
            "CreateTime": created}


def patched(ec2):
    fake_boto3 = mock.Mock()
    fake_boto3.client.return_value = ec2
    return mock.patch.object(deploy, "_boto3", return_value=fake_boto3)


def devices(request):
    return {
        mapping["DeviceName"]: mapping["Ebs"]
        for mapping in request["BlockDeviceMappings"]
    }


class Inf2ScaffoldTests(unittest.TestCase):
    def test_user_data_quotes_values_and_has_no_token(self):
        rendered = deploy.render_user_data(
            config(model_id="model with spaces; unsafe", hf_secret_id="secret/name")
        )
        self.assertIn("'model with spaces; unsafe'", rendered)
        self.assertIn("aws secretsmanager get-secret-value", rendered)
        self.assertNotIn("hf_actual_secret_value", rendered)
        self.assertIn('HF_TOKEN="$(cat /run/gemma4-hf-token)"', rendered)
        self.assertNotIn("__MODEL_ID__", rendered)

    def test_refuses_second_tagged_host(self):
        ec2 = FakeEc2([{"InstanceId": "i-existing"}])
        with patched(ec2):
            with self.assertRaisesRegex(RuntimeError, "i-existing"):
                deploy.launch(config(), apply=True)
        self.assertIsNone(ec2.run_request)

    def test_plan_does_not_launch(self):
        ec2 = FakeEc2()
        with patched(ec2):
            result = deploy.launch(config(), apply=False)
        self.assertEqual(result["action"], "plan")
        self.assertIsNone(ec2.run_request)

    def test_launch_requires_networking_arguments(self):
        ec2 = FakeEc2()
        with patched(ec2):
            with self.assertRaisesRegex(ValueError, "--subnet-id"):
                deploy.launch(config(subnet_id=None), apply=False)

    def test_apply_enforces_imdsv2_and_spot_choice(self):
        ec2 = FakeEc2()
        with patched(ec2):
            result = deploy.launch(config(market_type="spot"), apply=True)
        self.assertEqual(result["instance_id"], "i-new")
        self.assertEqual(ec2.run_request["MetadataOptions"]["HttpTokens"], "required")
        self.assertEqual(
            ec2.run_request["InstanceMarketOptions"]["MarketType"], "spot"
        )

    def test_root_volume_is_disposable_and_cache_volume_is_attached_separately(self):
        ec2 = FakeEc2()
        with patched(ec2):
            result = deploy.launch(config(), apply=True)
        mappings = devices(ec2.run_request)
        # A retained root volume is a pure cost leak: nothing ever reattaches it.
        self.assertTrue(mappings[deploy.ROOT_DEVICE]["DeleteOnTermination"])
        # RunInstances can only CREATE volumes, never attach an existing one, so
        # the cache volume must not be described here — that is what silently
        # minted a new volume per launch while the old one billed unattached.
        self.assertNotIn(deploy.CACHE_DEVICE, mappings)
        self.assertTrue(ec2.created_volume["Encrypted"])
        self.assertEqual(ec2.attached["VolumeId"], "vol-created")
        self.assertEqual(ec2.attached["Device"], deploy.CACHE_DEVICE)
        self.assertEqual(result["cache_volume_attached"], "vol-created")

    def test_reuse_cache_attaches_the_existing_volume(self):
        """The startup win: no new volume, so the checkpoint and NEFFs survive."""
        ec2 = FakeEc2(volumes=[volume("vol-warm")])
        with patched(ec2):
            result = deploy.launch(config(reuse_cache=True), apply=True)
        self.assertIsNone(ec2.created_volume)
        self.assertEqual(ec2.attached["VolumeId"], "vol-warm")
        self.assertEqual(result["cache_volume"]["action"], "reuse")

    def test_reuse_cache_falls_back_to_a_fresh_volume(self):
        ec2 = FakeEc2(volumes=[])
        with patched(ec2):
            result = deploy.launch(config(reuse_cache=True), apply=False)
        self.assertEqual(result["cache_volume"]["action"], "create")

    def test_ambiguous_reusable_volumes_are_an_error(self):
        """Attaching the wrong cache is worse than a cold start: it looks fine."""
        ec2 = FakeEc2(volumes=[volume("vol-a"), volume("vol-b")])
        with patched(ec2):
            with self.assertRaisesRegex(RuntimeError, "vol-a"):
                deploy.launch(config(reuse_cache=True), apply=False)

    def test_cache_volume_in_the_wrong_az_is_refused(self):
        """EBS is AZ-locked; a mismatch must name both AZs, not fail obscurely."""
        ec2 = FakeEc2(volumes=[volume("vol-elsewhere", az="us-east-1c")],
                      az="us-east-1a")
        with patched(ec2):
            with self.assertRaisesRegex(RuntimeError, "us-east-1c"):
                deploy.launch(config(cache_volume_id="vol-elsewhere"), apply=False)

    def test_attached_cache_volume_is_refused(self):
        ec2 = FakeEc2(volumes=[volume("vol-busy", state="in-use")])
        with patched(ec2):
            with self.assertRaisesRegex(RuntimeError, "in-use"):
                deploy.launch(config(cache_volume_id="vol-busy"), apply=False)

    def test_auto_discovered_ami_is_flagged_as_unpinned(self):
        ec2 = FakeEc2()
        with patched(ec2):
            explicit = deploy.launch(config(), apply=False)
        self.assertEqual(explicit["ami_source"], "explicit")

    def test_terminate_plan_reports_retained_cache_volume(self):
        host = {
            "InstanceId": "i-existing",
            "BlockDeviceMappings": [
                {"DeviceName": deploy.ROOT_DEVICE, "Ebs": {"VolumeId": "vol-root"}},
                {"DeviceName": deploy.CACHE_DEVICE, "Ebs": {"VolumeId": "vol-cache"}},
            ],
        }
        ec2 = FakeEc2([host])
        with patched(ec2):
            result = deploy.terminate(config(), apply=False)
        self.assertEqual(result["action"], "plan")
        self.assertEqual(result["targets"][0]["cache_volume_id"], "vol-cache")
        self.assertEqual(result["cache_volumes"], "retain")
        self.assertIsNone(ec2.terminated)

    def test_terminate_can_delete_the_cache_volume(self):
        """Cleanup has to be a supported action, or orphans accumulate silently."""
        host = {
            "InstanceId": "i-existing",
            "BlockDeviceMappings": [
                {"DeviceName": deploy.CACHE_DEVICE, "Ebs": {"VolumeId": "vol-cache"}},
            ],
        }
        ec2 = FakeEc2([host])
        with patched(ec2):
            result = deploy.terminate(config(delete_cache=True), apply=True)
        self.assertEqual(ec2.deleted_volumes, ["vol-cache"])
        self.assertEqual(result["deleted_volumes"], ["vol-cache"])

    def test_terminate_retains_the_cache_volume_by_default(self):
        host = {
            "InstanceId": "i-existing",
            "BlockDeviceMappings": [
                {"DeviceName": deploy.CACHE_DEVICE, "Ebs": {"VolumeId": "vol-cache"}},
            ],
        }
        ec2 = FakeEc2([host])
        with patched(ec2):
            deploy.terminate(config(), apply=True)
        self.assertEqual(ec2.deleted_volumes, [])

    def test_terminate_apply_terminates_only_tagged_hosts(self):
        ec2 = FakeEc2([{"InstanceId": "i-existing"}])
        with patched(ec2):
            result = deploy.terminate(config(), apply=True)
        self.assertEqual(result["state"], "shutting-down")
        self.assertEqual(ec2.terminated["InstanceIds"], ["i-existing"])

    def test_terminate_without_a_host_is_an_error(self):
        ec2 = FakeEc2()
        with patched(ec2):
            with self.assertRaisesRegex(RuntimeError, "No active EC2 host"):
                deploy.terminate(config(), apply=True)
        self.assertIsNone(ec2.terminated)


class NeuronEntrypointTests(unittest.TestCase):
    def test_pallas_is_refused_by_the_engine_not_routed_to_the_interpreter(self):
        """The entrypoint must NOT set JAX_E_PALLAS_INTERPRET.

        It used to, on the reasoning that the interpreter is better than a hard
        failure. It is not: the Pallas interpreter traces the fused W4A16 kernel
        body into the enclosing graph, unrolling its K loop per tile, so the
        "fallback" silently produces a far worse graph than the reference path.
        `ports/gemma4/backend.py` now reports that Neuron has no Pallas backend
        and `set_w4a16_impl` raises on "fused", which fails loudly at
        configuration time instead of quietly at serving time.
        """
        with mock.patch.dict(os.environ, {}, clear=True):
            entrypoint.configure_neuron()
            self.assertNotIn("JAX_E_PALLAS_INTERPRET", os.environ)
            # "neuron,cpu": quantize_ple_table needs a host device for the
            # 4.70 GB per-layer embedding table.
            self.assertEqual(os.environ["JAX_PLATFORMS"], "neuron,cpu")
            self.assertEqual(os.environ["JAX_DEFAULT_PRNG_IMPL"], "rbg")

    def test_entrypoint_does_not_pin_the_engine_platform(self):
        """JAX_E_PLATFORM is a testing override; on a real host it must be unset
        so detection reads the actual PJRT device."""
        with mock.patch.dict(os.environ, {}, clear=True):
            entrypoint.configure_neuron()
            self.assertNotIn("JAX_E_PLATFORM", os.environ)

    def test_configure_neuron_respects_a_preset_platform(self):
        with mock.patch.dict(os.environ, {"JAX_PLATFORMS": "cpu"}, clear=True):
            entrypoint.configure_neuron()
            self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")


@unittest.skipIf(shutil.which("bash") is None, "bash not available")
class UserDataShellTests(unittest.TestCase):
    def rendered(self):
        return deploy.render_user_data(config())

    def test_rendered_user_data_is_valid_bash(self):
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
            handle.write(self.rendered())
            path = handle.name
        try:
            result = subprocess.run(
                ["bash", "-n", path], capture_output=True, text=True
            )
        finally:
            os.unlink(path)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_token_is_handed_to_the_service_user(self):
        text = self.rendered()
        # ExecStartPre=+ runs as root; ExecStart runs as ubuntu. Without the
        # chown the unit crash-loops on a permission error it never logs well.
        self.assertIn("ExecStartPre=+/usr/local/bin/gemma4-fetch-hf-token", text)
        self.assertIn('chown ubuntu:ubuntu "$tmp"', text)
        self.assertIn("User=ubuntu", text)

    def test_swap_is_provisioned_before_the_memory_hungry_steps(self):
        text = self.rendered()
        # The 16 GiB inf2.xlarge host OOM-kills the Neuron graph load without
        # swap, taking the SSM agent with it -- and there is no inbound SSH.
        swap = text.index("mkswap /swapfile")
        self.assertLess(swap, text.index("pip install"))
        self.assertLess(swap, text.index("systemctl enable"))
        self.assertIn("'/swapfile none swap sw 0 0'", text)

    def test_swap_can_be_disabled_on_a_large_host(self):
        self.assertIn('SWAP_GIB=0\n', deploy.render_user_data(config(swap_gib=0)))

    def test_bootstrap_waits_for_a_reattached_cache_volume(self):
        """deploy.py attaches a reused volume after run_instances returns.

        The old code looked for the device exactly once. On a fast boot it found
        nothing, warned, and put the 9.6 GB checkpoint plus the Neuron cache on
        the root volume that dies at termination — losing the entire saving to a
        warning nobody reads.
        """
        text = self.rendered()
        self.assertIn("CACHE_WAIT_SECS=", text)
        wait = text.index("deadline=$(( SECONDS + CACHE_WAIT_SECS ))")
        self.assertLess(wait, text.index("blkid"))

    def test_cache_volume_is_never_reformatted_when_it_holds_data(self):
        """mkfs on a warm cache would silently destroy the whole point of it."""
        text = self.rendered()
        self.assertIn('blkid "$cache_dev" >/dev/null 2>&1 || mkfs.ext4', text)

    def test_bootstrap_phases_are_skippable_so_a_retry_is_cheap(self):
        text = self.rendered()
        self.assertIn("phase_done()", text)
        for phase in ("os-packages", "python-deps", "neuron-probe"):
            self.assertIn(f"phase_done {phase}", text)
        # A bare `mount` of an already-mounted path exits non-zero, and under
        # `set -e` that aborted the script — which made a retry impossible.
        self.assertIn('mountpoint -q "$CACHE_ROOT" || mount "$CACHE_ROOT"', text)

    def test_the_source_bundle_is_never_phase_skipped(self):
        """It is the one thing that changes between runs; skipping serves stale code."""
        text = self.rendered()
        self.assertNotIn("phase_done source", text)
        self.assertIn("aws s3 cp", text)

    def test_probe_gates_the_bootstrap_before_the_expensive_steps(self):
        """A broken stack should cost a minute, not the download plus a compile."""
        text = self.rendered()
        probe = text.index("jax_neuron/probe.py")
        self.assertLess(probe, text.index("systemctl enable"))
        self.assertIn("FATAL: the JAX Neuron stack does not work", text)

    def test_probe_and_service_share_one_path_definition(self):
        """Two PATH strings would drift, and the drift only shows at first compile."""
        text = self.rendered()
        # No venv, per CLAUDE.md: packages install into the DLAMI's own
        # interpreter, so neuronx-cc's console script lands in /usr/local/bin.
        # This asserted a /opt/gemma4/venv/bin prefix until the venv was removed.
        self.assertIn("SERVICE_PATH=/opt/aws/neuron/bin:", text)
        self.assertIn("/usr/local/bin", text)
        self.assertNotIn("/opt/gemma4/venv", text)
        self.assertIn("PATH=$SERVICE_PATH", text)
        self.assertIn('PATH="$SERVICE_PATH"', text)

    def test_compiler_flags_reach_the_service_environment(self):
        rendered = deploy.render_user_data(
            config(neuron_cc_flags="--model-type=transformer --target=inf2")
        )
        self.assertIn(
            "NEURON_CC_FLAGS_VALUE='--model-type=transformer --target=inf2'", rendered
        )
        self.assertIn("NEURON_CC_FLAGS=$NEURON_CC_FLAGS_VALUE", rendered)

    def test_entrypoint_defers_to_the_deployed_compiler_flags(self):
        # user_data puts NEURON_CC_FLAGS in the unit's EnvironmentFile, so the
        # entrypoint must not clobber it -- setdefault, never assignment.
        with mock.patch.dict(os.environ, {"NEURON_CC_FLAGS": "--target=inf2"}, clear=True):
            entrypoint.configure_neuron()
            self.assertEqual(os.environ["NEURON_CC_FLAGS"], "--target=inf2")

    def test_cache_volume_is_mounted_before_cache_dirs_are_created(self):
        text = self.rendered()
        self.assertLess(text.index('mount "$CACHE_ROOT"'), text.index("/huggingface"))
        # A reattached volume already holds the compile caches.
        self.assertIn('blkid "$cache_dev"', text)


if __name__ == "__main__":
    unittest.main()
