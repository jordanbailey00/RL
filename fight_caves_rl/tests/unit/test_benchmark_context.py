from fight_caves_rl.benchmarks.common import detect_host_class


def test_detect_host_class_for_wsl():
    host_class, is_wsl = detect_host_class(
        system_name="Linux",
        release_name="6.6.87.2-microsoft-standard-WSL2",
        version_name="#1 SMP PREEMPT_DYNAMIC",
        platform_string="Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
    )

    assert host_class == "wsl2"
    assert is_wsl is True


def test_detect_host_class_for_native_linux():
    host_class, is_wsl = detect_host_class(
        system_name="Linux",
        release_name="6.8.0-31-generic",
        version_name="#31-Ubuntu SMP",
        platform_string="Linux-6.8.0-31-generic-x86_64-with-glibc2.39",
    )

    assert host_class == "linux_native"
    assert is_wsl is False
