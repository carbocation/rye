use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn run(binary: &str, repository: &Path, prefix: &Path, threads: usize) -> Output {
    Command::new(binary)
        .arg(format!(
            "--eigenvec={}",
            repository.join("examples/example.eigenvec").display()
        ))
        .arg(format!(
            "--eigenval={}",
            repository.join("examples/example.eigenval").display()
        ))
        .arg(format!(
            "--pop2group={}",
            repository.join("examples/pop2group.txt").display()
        ))
        .arg(format!("--output={}", prefix.display()))
        .arg(format!("--threads={threads}"))
        .args(["--rounds=3", "--iter=20", "--attempts=4", "--seed=2026"])
        .output()
        .expect("run standalone rye")
}

fn output_path(prefix: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{suffix}", prefix.display()))
}

#[test]
fn output_is_independent_of_worker_count() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    let directory = std::env::temp_dir().join(format!("rye-cli-{}-{unique}", std::process::id()));
    fs::create_dir(&directory).expect("create test output directory");
    let serial_prefix = directory.join("serial");
    let parallel_prefix = directory.join("parallel");

    let serial = run(env!("CARGO_BIN_EXE_rye"), &repository, &serial_prefix, 1);
    assert!(
        serial.status.success(),
        "serial run failed:\n{}\n{}",
        String::from_utf8_lossy(&serial.stdout),
        String::from_utf8_lossy(&serial.stderr)
    );
    let parallel = run(env!("CARGO_BIN_EXE_rye"), &repository, &parallel_prefix, 4);
    assert!(
        parallel.status.success(),
        "parallel run failed:\n{}\n{}",
        String::from_utf8_lossy(&parallel.stdout),
        String::from_utf8_lossy(&parallel.stderr)
    );

    let serial_q = fs::read(output_path(&serial_prefix, "-20.7.Q")).expect("read serial Q");
    let parallel_q = fs::read(output_path(&parallel_prefix, "-20.7.Q")).expect("read parallel Q");
    assert_eq!(serial_q, parallel_q);
    let serial_fam = fs::read(output_path(&serial_prefix, "-20.fam")).expect("read serial FAM");
    let parallel_fam =
        fs::read(output_path(&parallel_prefix, "-20.fam")).expect("read parallel FAM");
    assert_eq!(serial_fam, parallel_fam);
    assert!(serial_q.starts_with(b"European\tAsian\tAmerindian\t"));
    assert_eq!(
        serial_fam.iter().filter(|&&byte| byte == b'\n').count(),
        3_400
    );

    fs::remove_dir_all(directory).expect("remove test output directory");
}
