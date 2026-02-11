//! GPU-accelerated MSM for Pallas curve using pallas-gpu crate

use crate::provider::pasta::pallas;
use ff::{Field, PrimeField};
use halo2curves::{group::Group, CurveAffine};

/// Safely extract a field element as [u64; 4] in standard (non-Montgomery) form
/// Uses to_repr() which converts from internal Montgomery form to canonical bytes
#[inline]
fn field_to_standard_limbs<F: PrimeField>(f: &F) -> [u64; 4] {
    let repr = f.to_repr();
    let bytes = repr.as_ref();
    [
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
    ]
}

/// Safely reconstruct a field element from [u64; 4] in standard (non-Montgomery) form
/// Uses from_repr() which converts from canonical bytes to internal Montgomery form
/// Returns None if the value is >= modulus (invalid field element)
#[inline]
fn standard_limbs_to_field<F: PrimeField>(limbs: &[u64; 4]) -> Option<F> {
    let mut repr = F::Repr::default();
    let bytes: &mut [u8] = repr.as_mut();
    bytes[0..8].copy_from_slice(&limbs[0].to_le_bytes());
    bytes[8..16].copy_from_slice(&limbs[1].to_le_bytes());
    bytes[16..24].copy_from_slice(&limbs[2].to_le_bytes());
    bytes[24..32].copy_from_slice(&limbs[3].to_le_bytes());
    F::from_repr(repr).into()
}

/// Perform GPU-accelerated multi-scalar multiplication for Pallas curve
///
/// All data is passed to the GPU in standard (non-Montgomery) form.
/// The GPU converts points to Montgomery form internally before doing arithmetic,
/// and converts the result back to standard form before returning.
pub fn vartime_multiscalar_mul(
    scalars: &[pallas::Scalar],
    bases: &[pallas::Affine],
) -> pallas::Point {
    use pallas_gpu::msm_u64;

    if scalars.is_empty() || bases.is_empty() {
        return pallas::Point::identity();
    }

    assert_eq!(
        scalars.len(),
        bases.len(),
        "scalars and bases must have same length"
    );

    // Convert scalars to standard form 64-bit limbs
    // GPU extracts bit windows from scalars, so standard form is correct
    let scalar_limbs: Vec<[u64; 4]> = scalars
        .iter()
        .map(|s| field_to_standard_limbs(s))
        .collect();

    // Convert point coordinates to standard form 64-bit limbs
    // GPU will convert these to Montgomery form internally before arithmetic
    let point_limbs: Vec<([u64; 4], [u64; 4])> = bases
        .iter()
        .map(|p| {
            let coords = p.coordinates().unwrap();
            let x_limbs = field_to_standard_limbs(coords.x());
            let y_limbs = field_to_standard_limbs(coords.y());
            (x_limbs, y_limbs)
        })
        .collect();

    // Call GPU MSM
    // Result comes back in standard form (GPU converts from Montgomery before returning)
    let (result_x, result_y, result_z) = msm_u64(&scalar_limbs, &point_limbs)
        .expect("GPU MSM failed");

    eprintln!("[GPU DEBUG] raw result X: {:016x?}", result_x);
    eprintln!("[GPU DEBUG] raw result Y: {:016x?}", result_y);
    eprintln!("[GPU DEBUG] raw result Z: {:016x?}", result_z);

    // Convert result back to pallas field elements
    let x: Option<pallas::Base> = standard_limbs_to_field(&result_x);
    let y: Option<pallas::Base> = standard_limbs_to_field(&result_y);
    let z: Option<pallas::Base> = standard_limbs_to_field(&result_z);

    eprintln!("[GPU DEBUG] from_repr X valid: {}", x.is_some());
    eprintln!("[GPU DEBUG] from_repr Y valid: {}", y.is_some());
    eprintln!("[GPU DEBUG] from_repr Z valid: {}", z.is_some());

    let x = x.expect("GPU result X is not a valid field element");
    let y = y.expect("GPU result Y is not a valid field element");
    let z = z.expect("GPU result Z is not a valid field element");

    // Check for identity (Z = 0)
    if z.is_zero().into() {
        eprintln!("[GPU DEBUG] result is identity (Z=0)");
        return pallas::Point::identity();
    }

    // Convert from projective (X:Y:Z) to affine (x/z, y/z) then to projective point
    let z_inv = z.invert().expect("Z should be invertible (non-zero)");
    let affine_x = x * z_inv;
    let affine_y = y * z_inv;

    eprintln!("[GPU DEBUG] affine x: {:?}", field_to_standard_limbs(&affine_x));
    eprintln!("[GPU DEBUG] affine y: {:?}", field_to_standard_limbs(&affine_y));

    // Check if point is on curve before unwrapping
    let on_curve = pallas::Affine::from_xy(affine_x, affine_y);
    if Option::<pallas::Affine>::from(on_curve).is_none() {
        eprintln!("[GPU DEBUG] ERROR: point NOT on curve y^2 = x^3 + 5");
        // Verify: compute y^2 and x^3 + 5
        let y_sq = affine_y * affine_y;
        let x_cu = affine_x * affine_x * affine_x;
        let rhs = x_cu + pallas::Base::from(5u64);
        eprintln!("[GPU DEBUG] y^2     = {:016x?}", field_to_standard_limbs(&y_sq));
        eprintln!("[GPU DEBUG] x^3 + 5 = {:016x?}", field_to_standard_limbs(&rhs));
        panic!("GPU MSM returned a point not on the Pallas curve");
    }

    let affine = Option::<pallas::Affine>::from(on_curve).unwrap();
    pallas::Point::from(affine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::msm::msm;
    use ff::Field;
    use halo2curves::group::{Curve, Group};
    use rand::rngs::OsRng;

    #[test]
    fn test_gpu_msm_small() {
        let n = 10;
        let scalars: Vec<pallas::Scalar> = (0..n)
            .map(|_| pallas::Scalar::random(&mut OsRng))
            .collect();
        let bases: Vec<pallas::Affine> = (0..n)
            .map(|_| pallas::Point::random(&mut OsRng).to_affine())
            .collect();

        let cpu_result = msm(&scalars, &bases);
        let gpu_result = vartime_multiscalar_mul(&scalars, &bases);

        assert_eq!(
            cpu_result.to_affine(),
            gpu_result.to_affine(),
            "GPU result should match CPU result"
        );
    }

    #[test]
    fn test_gpu_msm_medium() {
        let n = 1000;
        let scalars: Vec<pallas::Scalar> = (0..n)
            .map(|_| pallas::Scalar::random(&mut OsRng))
            .collect();
        let bases: Vec<pallas::Affine> = (0..n)
            .map(|_| pallas::Point::random(&mut OsRng).to_affine())
            .collect();

        let cpu_result = msm(&scalars, &bases);
        let gpu_result = vartime_multiscalar_mul(&scalars, &bases);

        assert_eq!(
            cpu_result.to_affine(),
            gpu_result.to_affine(),
            "GPU result should match CPU result for 1000 points"
        );
    }

    #[test]
    fn bench_cpu_vs_gpu() {
        use std::io::Write;
        use std::time::Instant;

        let sizes = [100_000, 200_000, 500_000, 1_000_000];
        let warmup = 2;
        let runs = 5;

        let mut out = Vec::new();
        out.push(format!("{:<12} {:>12} {:>12} {:>10}", "Points", "CPU (ms)", "GPU (ms)", "Speedup"));
        out.push("-".repeat(50));

        for &n in &sizes {
            eprintln!("Benchmarking n={}...", n);
            let scalars: Vec<pallas::Scalar> = (0..n).map(|_| pallas::Scalar::random(&mut OsRng)).collect();
            let bases: Vec<pallas::Affine> = (0..n).map(|_| pallas::Point::random(&mut OsRng).to_affine()).collect();

            for _ in 0..warmup {
                let _ = msm(&scalars, &bases);
                let _ = vartime_multiscalar_mul(&scalars, &bases);
            }

            let mut cpu_ms = Vec::new();
            for _ in 0..runs {
                let t = Instant::now();
                let _ = msm(&scalars, &bases);
                cpu_ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }

            let mut gpu_ms = Vec::new();
            for _ in 0..runs {
                let t = Instant::now();
                let _ = vartime_multiscalar_mul(&scalars, &bases);
                gpu_ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }

            let cpu_result = msm(&scalars, &bases);
            let gpu_result = vartime_multiscalar_mul(&scalars, &bases);
            assert_eq!(cpu_result.to_affine(), gpu_result.to_affine(), "Mismatch at n={}", n);

            let ca = cpu_ms.iter().sum::<f64>() / runs as f64;
            let ga = gpu_ms.iter().sum::<f64>() / runs as f64;
            out.push(format!("{:<12} {:>12.3} {:>12.3} {:>9.2}x", n, ca, ga, ca / ga));
        }

        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        let path = format!("{}/msm_benchmark.txt", home);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "=== Pallas MSM Benchmark: CPU vs GPU ===").unwrap();
        writeln!(f, "Warmup: {}, Runs: {}\n", warmup, runs).unwrap();
        for line in &out { writeln!(f, "{}", line).unwrap(); }
        writeln!(f, "\nAll results verified correct (CPU == GPU).").unwrap();
        eprintln!("\nResults written to {}", path);
    }
}
