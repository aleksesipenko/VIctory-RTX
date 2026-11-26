use super::oracle;
use victory_shader::rt_packet::*;

const IDENTITY: Transform = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]];
fn instance(blas: u32, id: u32, mask: u8, transform: Transform) -> Instance {
    Instance {
        blas,
        id,
        mask,
        transform,
        contribution: id + 0x1500,
        front_ccw: false,
    }
}
fn ray(x: f32, y: f32, mask: u8) -> Ray {
    Ray {
        origin: [x, y, 0.],
        direction: [0., 0., 1.],
        t_min: 0.,
        t_max: 100.,
        mask,
    }
}
fn quad(x: f32, y: f32, z: f32) -> [Triangle; 2] {
    let a = [x, y, z];
    let b = [x + 1., y, z];
    let c = [x, y + 1., z];
    let d = [x + 1., y + 1., z];
    // Rotate both vertex lists to force a non-identity I/J swizzle in the pair.
    [[b, c, a], [c, b, d]]
}

pub fn small(layers: usize) -> Packet {
    // Farther triangle comes first in BLAS input order and overlaps the
    // first half of the near quad. Geometry and primitive IDs are observable.
    let mut second = instance(
        0,
        0x123456,
        0x15,
        [[-2., 0., 0., 4.], [0., 0.5, 0., 1.], [0., 0., 1.5, 1.]],
    );
    second.front_ccw = true;
    let mut masked = IDENTITY;
    masked[0][3] = -8.;
    let mirror = instance(
        0,
        0x61,
        0xff,
        [[-1., 0., 0., -4.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
    );
    let mut tilted = instance(
        0,
        0x62,
        0xd5,
        [[0., -1., 0., -2.], [2., 0., 0., 3.], [0.25, 0., 1., -1.]],
    );
    tilted.front_ccw = true;
    let mut tlases = vec![vec![
        instance(0, 0x31, 0xc0, IDENTITY),
        second,
        instance(0, 0x99, 0, masked),
        mirror,
        tilted,
    ]];
    for n in 1..layers {
        let mut m = IDENTITY;
        m[0][3] = 8. * n as f32;
        m[2][3] = n as f32;
        // Also overlaps Near at x=0. Near wins there despite this closer surface.
        let mut overlap = IDENTITY;
        overlap[2][3] = -2.;
        tlases.push(vec![
            instance(0, 0x40 + n as u32, 0xff, m),
            instance(0, 0x50 + n as u32, 0xc0, overlap),
        ]);
    }
    let rays = vec![
        ray(0.125, 0.25, 0xc0),
        ray(0.875, 0.625, 0xd5),
        ray(0.25, 0.75, 0xff),
        ray(0.125, 0.25, 0),
        ray(0.125, 0.25, 0x15),
        ray(3.75, 1.125, 0xd5),
        ray(2.375, 1.375, 0xc0),
        ray(2.375, 1.375, 0x15),
        ray(8.25, 0.375, 0xff),
        ray(16.25, 0.375, 0xd5),
        ray(-20., -1., 0xff),
        Ray {
            t_min: 4.,
            ..ray(0.125, 0.25, 0xc0)
        },
        Ray {
            t_max: 2.5,
            ..ray(0.125, 0.25, 0xc0)
        },
        Ray {
            origin: [0.125, 0.25, 12.],
            direction: [0., 0., -2.],
            ..ray(0., 0., 0xc0)
        },
        Ray {
            direction: [0.03125, 0., 2.],
            ..ray(0.0625, 0.25, 0xd5)
        },
        ray(-7.875, 0.25, 0xff),
        ray(-4.125, 0.25, 0xff),
        ray(-2.25, 3.25, 0xd5),
        ray(-2.25, 3.25, 0x20),
    ];
    let mut p = Packet {
        blases: vec![vec![
            Geometry {
                opaque: true,
                triangles: vec![quad(0., 0., 8.)[0]],
            },
            Geometry {
                opaque: false,
                triangles: quad(0., 0., 3.).to_vec(),
            },
        ]],
        tlases,
        rays: rays.into_iter().map(|r| (r, vec![])).collect(),
    };
    oracle::freeze(&mut p);
    p
}

pub fn workload() -> Packet {
    let blases = (0..2)
        .map(|b| {
            let triangles = (0..32)
                .flat_map(|y| {
                    (0..32).flat_map(move |x| {
                        let z = 3. + ((x * 3 + y * 5 + b * 7) % 17) as f32 * 0.0625;
                        quad(x as f32, y as f32, z)
                    })
                })
                .collect();
            vec![Geometry {
                opaque: true,
                triangles,
            }]
        })
        .collect();
    let instances = (0..4)
        .map(|n| {
            let mut m = IDENTITY;
            m[0][3] = (n % 2) as f32 * 32.;
            m[1][3] = (n / 2) as f32 * 32.;
            instance(n % 2, n + 17, if n % 2 == 0 { 0xc0 } else { 0x15 }, m)
        })
        .collect();
    let rays = (0..256 * 256)
        .map(|n| {
            let x = (n % 256) as f32 * 0.25 + 0.0625;
            let y = (n / 256) as f32 * 0.25 + 0.09375;
            let mut r = ray(x, y, if n % 13 == 0 { 0xc0 } else { 0xd5 });
            r.direction = [0.0078125, -0.00390625, 1.];
            (r, vec![])
        })
        .collect();
    let mut p = Packet {
        blases,
        tlases: vec![instances],
        rays,
    };
    oracle::freeze(&mut p);
    p
}

/// Both instances have identical world bounds but opposite depth slopes.
/// A BVH visit order cannot stand in for comparing intersection distances.
pub fn crossed_obstacles() -> Packet {
    let mut tris = quad(0., 0., 0.);
    for p in tris.iter_mut().flatten() {
        p[2] = 2. + 8. * p[0];
    }
    let tlases = vec![vec![
        instance(0, 11, 0xff, IDENTITY),
        instance(
            0,
            22,
            0xff,
            [[-1., 0., 0., 1.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
        ),
    ]];
    let rays = (0..32)
        .map(|n| (ray((n as f32 + 0.5) / 32., 0.3125, 0xff), Vec::new()))
        .collect();
    let mut p = Packet {
        blases: vec![vec![Geometry {
            opaque: true,
            triangles: tris.to_vec(),
        }]],
        tlases,
        rays,
    };
    oracle::freeze(&mut p);
    p
}
