extern crate rand;
use rand::{thread_rng, Rng};

use std::fs;
use std::vec::Vec;
use std::ops::{Add, Sub, Mul, Div, Neg, Index};
use std::io::{BufWriter, Write};
use std::option::Option;
use std::cmp;
use std::thread;
//use num_traits::pow;
//use std::Borrow::BorrowMut;

#[derive(Debug, Copy, Clone)]
pub struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    pub fn new(p:(f64, f64, f64)) -> Vec3 {
        Vec3{x:p.0, y:p.1, z:p.2}
    }
    pub fn addc(self, d: f64) -> Vec3{
        Vec3{x:self.x + d, y:self.y + d, z:self.z + d}
    }

    pub fn subc(self, d: f64) -> Vec3{
        Vec3{x:self.x - d, y:self.y - d, z:self.z - d}
    }

    pub fn mulc(self, d: f64) -> Vec3{
        Vec3{x:self.x * d, y:self.y * d, z:self.z * d}
    }

    pub fn divc(self, d: f64) -> Vec3{
        //print!("{}\n", self.x / d);
        Vec3{x:self.x / d, y:self.y / d, z:self.z / d}
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self{x: 0.0, y:0.0, z:0.0}
    }
}

pub fn dot(v1: Vec3, v2: Vec3) -> f64 {
    v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
}
    
pub fn cross(v1: Vec3, v2: Vec3) -> Vec3 {
    Vec3 {
        x: v1.y * v2.z - v1.z * v2.y,
        y: v1.z * v2.x - v1.x * v2.z,
        z: v1.x * v2.y - v1.y * v2.x,
    }
}

pub fn copysign(x: f64, y:f64) -> f64 {
    if y < 0.0 { -1.0 * x.abs() } else { x.abs() }
}

pub fn normalize(v: Vec3) -> Vec3 {
    let size = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    Vec3 {
        x: v.x / size,
        y: v.y / size,
        z: v.z / size,
    }
}

pub fn tonemap(v: f64) -> i32 {
    cmp::min(cmp::max((v.powf(1.0 / 2.2) * 255.0) as i32, 0), 255)
}

pub fn tangent_space(v: &Vec3) -> (Vec3, Vec3) {
    let s = copysign(1.0, v.z);
    let a = -1.0 / (s + v.z);
    let b = v.x * v.y * a;
    (Vec3{x:1.0 + s * v.x * v.x * a, y:s * b, z:-s * v.x},
     Vec3{x:b, y:s + v.y * v.y * a, z: -v.y})
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vec3{
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul for Vec3 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl Div for Vec3 {
    type Output = Vec3;
    fn div(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x / other.x,
            y: self.y / other.y,
            z: self.z / other.z,
        }
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}
impl<T> Index<T> for Vec3 {
    type Output = Vec3;
    fn index(&self, index: T) -> &Self::Output {
        return &self[index];
    }
}

#[derive(Debug, Copy, Clone)]
struct Ray {
    o: Vec3,
    d: Vec3,
}

#[derive(Debug, Copy, Clone)]
enum Material {
    Diffuse,
    Mirror,
    Fresnel,
}

#[derive(Debug, Copy, Clone)]
enum ObjectType {
    Sphere,
    Plane,
    Triangle,
}

#[derive(Debug, Copy, Clone)]
struct Object {
    tp: ObjectType,

    //parameters for sphere
    p: Vec3,
    r: f64,
    
    //parameters for plane
    n: Vec3, //norm
    c: Vec3, //const point

    //parameters for triangle
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    
    //parameters for material
    refl: Vec3, //Reflectance
    illu: Vec3, //illuminance
    mat: Material, //Material
    ior: f64,
}

impl Default for Object {
    fn default() -> Self {
        Self{tp: ObjectType::Sphere, 
            p:Default::default(), r:1.0,
            n:Default::default(), c:Default::default(),
            p0:Default::default(), p1:Default::default(), p2:Default::default(),
            refl:Vec3::new((0.75, 0.75, 0.75)), illu:Default::default(), mat:Material::Diffuse, ior:0.0
        }
    }
}

impl Object {
    pub fn intersect(&self, ray: &Ray, tmin: &f64, tmax: &f64) -> Option<Hit> {
        match self.tp {
            ObjectType::Sphere => {
                let op = self.p - ray.o;
                let b = dot(op, ray.d);
                let det = b * b - dot(op, op) + self.r * self.r;
                if det < 0.0 { return None; }
                let t1 = b - det.sqrt();
                if tmin < &t1 && &t1 < tmax {
                    return Some(Hit{t:t1, p:Default::default(), n:Default::default(), object:*self});
                }
                let t2 = b + det.sqrt();
                if tmin < &t2 && &t2 < tmax {
                    return Some(Hit{t:t2, p:Default::default(), n:Default::default(), object:*self});
                }
                    None
                }
            ObjectType::Plane => {
                let n = normalize(self.n);
                let t = -(dot(ray.o, n) - dot(self.c, n)) / dot(ray.d, n);
                if tmin < &t && &t < tmax {
                    return Some(Hit{t:t, p:Default::default(), n:Default::default(), object:*self});
                } else {
                    None
                }
            }
            ObjectType::Triangle => {
                let e1 = self.p1 - self.p0;
                let e2 = self.p2 - self.p0;
                let r = ray.o - self.p0;
                let u = cross(ray.d, e2);
                let v = cross(r, e1);
                let beta = dot(u, r) / dot(u, e1);
                let gamma = dot(v, ray.d) / dot(u, e1);
                if 0.0 < beta && 0.0 < gamma && beta + gamma < 1.0 {
                    let t = dot(v, e2) / dot(u, e1);
                    if tmin < &t && &t < tmax {
                        let n = normalize(cross(e1, e2));
                        return Some(Hit{t:t, p:Default::default(), n:n, object:*self});
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }
    pub fn mov(&self, p:Vec3) -> Object {
        match self.tp {
            ObjectType::Sphere => {
                let np = self.p + p;
                return Object{tp:ObjectType::Sphere, p:np, r:self.r, refl:self.refl, illu:self.illu, mat:self.mat, ior:self.ior,
                ..Default::default()};
            }
            ObjectType::Plane => {
                return *self;
            }
            ObjectType::Triangle => {
                let np0 = self.p0 + p;
                let np1 = self.p1 + p;
                let np2 = self.p2 + p;
                return Object{tp:ObjectType::Triangle, p0:np0, p1:np1, p2:np2, refl:self.refl, illu:self.illu, mat:self.mat, ior:self.ior,
                ..Default::default()};
            }
        }
    }
}



#[derive(Debug, Copy, Clone)]
struct Hit {
    t: f64,
    p: Vec3,
    n: Vec3,
    object: Object,
}

#[derive(Debug, Clone)]
struct Scene {
    objects: Vec<Object>,
}

impl Scene {
    pub fn intersect(self, ray: &Ray, tmin: &mut f64, tmax: &mut f64) -> Option<Hit> {
        let mut minh: Option<Hit> = None;
        let mut ret: Option<Hit> = None;
        for mut object in self.objects {
            let h = object.intersect(&ray, &tmin, &tmax);
            if h.is_none() {
                continue;
            }
            minh = h;
            *tmax = minh.unwrap().t;
        }
        if minh.is_some() {
            match minh.unwrap().object.tp {
                ObjectType::Sphere => {
                    let s = minh.unwrap().object;
                    //print!("{}\n", minh.unwrap().t);
                    ret = Some(Hit{t:minh.unwrap().t, p: ray.o + ray.d.mulc(minh.unwrap().t), n:Default::default(), object:s});
                    ret = Some(Hit{t:ret.unwrap().t, p:ret.unwrap().p, n: (ret.unwrap().p - s.p).divc(s.r), object:s});
                }
                ObjectType::Plane => {
                    let s = minh.unwrap().object;
                    ret = Some(Hit{t:minh.unwrap().t, p: ray.o + ray.d.mulc(minh.unwrap().t), n: s.n, object:s});
                }
                ObjectType::Triangle => {
                    let s = minh.unwrap().object;
                    let n = minh.unwrap().n;
                    //print!("{}\n", minh.unwrap().t);
                    ret = Some(Hit{t:minh.unwrap().t, p: ray.o + ray.d.mulc(minh.unwrap().t), n:n, object:s});
                }
            }
        }
        ret
    }
}

fn main()
{
    let pi = 3.141592653589;
    // Image size
    let w = 1024;
    let h = 768;
    // Camera Parameters
    let eye = Vec3{x:5.0, y:5.0, z:10.0};
    let center = Vec3{x:0.75, y:0.3, z:0.0};
    //let eye = Vec3{x:50.0, y:52.0, z:295.6};
    //let center = eye + Vec3{x:0.0, y:-0.042612, z:-1.0};
    let up = Vec3{x:0.0, y:1.0, z:0.0};
    let fov: f64 = pi / 6.0;
    let aspect = (w as f64) / (h as f64);
    let spp = 12000;
    // Basis vectors for camera coordinates
    let w_e = normalize(eye - center);
    let u_e = normalize(cross(up, w_e));
    let v_e = cross(w_e, u_e);

    let mut scene = Scene{objects: Vec::new()};
    //Scene 1

    scene.objects.push(Object{tp:ObjectType::Plane,
        p:Default::default(), r:0.0, n:Vec3{x:0.0, y:1.0, z:0.0}, c:Vec3{x:0.0, y:0.1, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Fresnel, ior:1.333, ..Default::default()});
    scene.objects.push(Object{tp:ObjectType::Plane,
        p:Default::default(), r:0.0, n:Vec3{x:0.0, y:0.0, z:1.0}, c:Vec3{x:0.0, y:0.0, z:-3.7},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Mirror, ior:0.0, ..Default::default()});
    scene.objects.push(Object{tp:ObjectType::Plane,
        p:Default::default(), r:0.0, n:Vec3{x:0.0, y:-1.0, z:0.0}, c:Vec3{x:0.0, y:5.5, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:1.0, y:1.0, z:1.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()});
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:1.2, y:-0.5, z:0.0}, r:1.0, n:Vec3{x:0.0, y:0.0, z:0.0}, c:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:1.0, y:1.0, z:1.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()});
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:1.2, y:4.5, z:0.0}, r:1.0, n:Vec3{x:0.0, y:0.0, z:0.0}, c:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:12.0, y:12.0, z:12.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()});
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:1.5, y:1.0, z:-2.0}, r:1.0, n:Vec3{x:0.0, y:0.0, z:0.0}, c:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.25, z:0.25}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()});
    /*scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:0.0, y:0.0, z:3.5}, p1:Vec3{x:0.0, y:0.0, z:-3.5}, p2:Vec3{x:0.0, y:(3.0 as f64).sqrt()*1.5, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Mirror, ior:0.0, ..Default::default()}.mov(Vec3{x:-1.5, y:0.5, z:0.0}));*/
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:-1.2, y:0.75, z:1.0}, r:0.75, n:Vec3{x:0.0, y:0.0, z:0.0}, c:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.25, y:0.75, z:0.25}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Mirror, ior:0.0, ..Default::default()});
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:0.5, y:0.5, z:2.0}, r:0.5, n:Vec3{x:0.0, y:0.0, z:0.0}, c:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.25, y:0.25, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Fresnel, ior:1.5168, ..Default::default()});
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:1.7, y:0.3, z:2.5}, r:0.3, n:Vec3{x:0.0, y:0.0, z:0.0}, c:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.25}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Mirror, ior:0.0, ..Default::default()});
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:2.7, y:0.2, z:2.5}, r:0.2, n:Vec3{x:0.0, y:0.0, z:0.0}, c:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.25, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Fresnel, ior:1.1, ..Default::default()});

    
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:-1.0 + 0.7, y:0.35, z:-1.5 + 0.5}, r:0.35, n:Vec3{x:0.0, y:0.0, z:0.0}, c:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.25, z:0.25}, illu:Vec3{x:1.0, y:1.0, z:4.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()});
    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:0.0, y:0.0, z:0.0}, p1:Vec3{x:1.0, y:0.0, z:0.0}, p2:Vec3{x:0.0, y:1.0, z:0.0},
        refl:Vec3{x:0.75, y:0.25, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Fresnel, ior:1.5, ..Default::default()}
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5})); 
    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:1.0, y:1.0, z:0.0}, p1:Vec3{x:0.0, y:1.0, z:0.0}, p2:Vec3{x:1.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.25, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Fresnel, ior:1.5, ..Default::default()}//Back
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5})); 
    /*scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:1.0, y:0.0, z:0.0}, p1:Vec3{x:1.0, y:0.0, z:1.0}, p2:Vec3{x:1.0, y:1.0, z:1.0},
        refl:Vec3{x:1.0, y:1.0, z:1.0}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Mirror, ior:0.0, ..Default::default()}
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5})); 
    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:1.0, y:1.0, z:0.0}, p1:Vec3{x:1.0, y:0.0, z:0.0}, p2:Vec3{x:1.0, y:1.0, z:1.0},
        refl:Vec3{x:1.0, y:1.0, z:1.0}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Mirror, ior:0.0, ..Default::default()} //Right
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5}));
    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:0.0, y:1.0, z:0.0}, p1:Vec3{x:1.0, y:1.0, z:0.0}, p2:Vec3{x:1.0, y:1.0, z:1.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Fresnel, ior:1.5, ..Default::default()}
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5})); 
    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:0.0, y:1.0, z:0.0}, p1:Vec3{x:1.0, y:1.0, z:1.0}, p2:Vec3{x:0.0, y:1.0, z:1.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Fresnel, ior:1.5, ..Default::default()}//Top
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5}));*/
    

    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:1.0, y:1.0, z:1.0}, p1:Vec3{x:1.0, y:0.0, z:1.0}, p2:Vec3{x:0.0, y:1.0, z:1.0},
        refl:Vec3{x:0.75, y:0.25, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Fresnel, ior:1.5, ..Default::default()}
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5})); 
    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:0.0, y:0.0, z:1.0}, p1:Vec3{x:0.0, y:1.0, z:1.0}, p2:Vec3{x:1.0, y:0.0, z:1.0},
        refl:Vec3{x:0.75, y:0.25, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Fresnel, ior:1.5, ..Default::default()}//Front
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5})); 
    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:0.0, y:0.0, z:1.0}, p1:Vec3{x:0.0, y:1.0, z:1.0}, p2:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Mirror, ior:0.0, ..Default::default()}
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5})); 
    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:0.0, y:1.0, z:1.0}, p1:Vec3{x:0.0, y:1.0, z:0.0}, p2:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Mirror, ior:0.0, ..Default::default()}//Left
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5})); 
    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:0.0, y:0.0, z:0.0}, p1:Vec3{x:0.0, y:0.0, z:1.0}, p2:Vec3{x:1.0, y:0.0, z:1.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Diffuse, ior:1.0, ..Default::default()}
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5})); 
    scene.objects.push(Object{tp:ObjectType::Triangle,
        p0:Vec3{x:0.0, y:0.0, z:0.0}, p1:Vec3{x:1.0, y:0.0, z:1.0}, p2:Vec3{x:1.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Diffuse, ior:1.0, ..Default::default()}//Bottom
        .mov(Vec3{x:-1.0, y:0.0, z:-1.5}));

    //scene.spheres.push(Sphere{p:Vec3{x:-0.5, y:0.0, z:0.0,}, r:1.0, refl:Vec3{x:1.0, y:0.0, z:0.0}});
    //scene.spheres.push(Sphere{p:Vec3{x: 0.5, y:0.0, z:0.0,}, r:1.0, refl:Vec3{x:0.0, y:1.0, z:0.0}});*/
    
    //Scene 2 - smallpt
    /*scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:1.0e+5 + 1.0, y:40.8, z:81.6}, r:1.0e+5, n:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.25, z:0.25}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()}); //Left
    scene.objects.push(Object{tp:ObjectType::Sphere,
    //scene.spheres.push(Sphere{p:Vec3{x:-0.5, y:0.0, z:0.0,}, r:1.0, refl:Vec3{x:1.0, y:0.0, z:0.0}});
    //scene.spheres.push(Sphere{p:Vec3{x: 0.5, y:0.0, z:0.0,}, r:1.0, refl:Vec3{x:0.0, y:1.0, z:0.0}});*/
    
    //Scene 2 - smallpt
    /*scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:1.0e+5 + 1.0, y:40.8, z:81.6}, r:1.0e+5, n:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.25, z:0.25}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()}); //Left
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:-1.0e+5 + 99.0, y:40.8, z:81.6}, r:1.0e+5, n:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.25, y:0.25, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()}); //Right
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:50.0, y:40.8, z:1.0e+5}, r:1.0e+5, n:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()}); //Back
    //scene.objects.push(Sphere{p:Vec3{x:50.0, y:40.8, z:-1.0e+5 + 170.0}, r:1.0e+5, refl:Vec3{x:0.999, y:0.999, z:0.999},
        //illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Diffuse, ior: 0.0}); //Front
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:50.0, y:1.0e+5, z:81.6}, r:1.0e+5, n:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()}); //Bottom
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:50.0, y:-1.0e+5 + 81.6, z:81.6}, r:1.0e+5, n:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.75, y:0.75, z:0.75}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()}); //Top
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:27.0, y:16.5, z:47.0}, r: 16.5, n:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.999, y:0.999, z:0.999}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Mirror, ior:0.0, ..Default::default()}); //Mirr
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:73.0, y:16.5, z:78.0}, r:16.5, n:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.999, y:0.999, z:0.999}, illu:Vec3{x:0.0, y:0.0, z:0.0}, mat:Material::Fresnel, ior:1.5168, ..Default::default()}); //Glas
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:48.0, y:10.0, z:100.0}, r:10.0, n:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.25, y:0.75, z:0.25}, illu:Vec3{x:0.1, y:0.5, z:0.1}, mat:Material::Diffuse, ior:0.0, ..Default::default()}); //Green Ball
    scene.objects.push(Object{tp:ObjectType::Sphere,
        p:Vec3{x:50.0, y:681.6-0.27, z:81.6}, r:600.0, n:Vec3{x:0.0, y:0.0, z:0.0},
        refl:Vec3{x:0.0, y:0.0, z:0.0}, illu:Vec3{x:12.0, y:12.0, z:12.0}, mat:Material::Diffuse, ior:0.0, ..Default::default()}); //Lite*/
    
    //Output File Header
    let mut header: String = "P3\n".to_string();
    header = header + &w.to_string() + &" ".to_string() + &h.to_string() + &"\n255\n".to_string();
    //let hit = b"255 0 255\n";
    //let nothit = b"0 0 0\n";
    let mut f = BufWriter::new(fs::File::create("result.ppm").unwrap());
    f.write(header.as_bytes()).unwrap();
    let mut img: Vec<Vec3> = Vec::new();
    let mut rng = thread_rng();
    //let mut children = vec![];
    //for thr in 0..4{
    //    children.push(thread::spawn(move||
    for i in 0..(w*h) {
        //let t = thread::spawn(move || {
        if i % 10000 == 0 {
            print!("{} %\n", ((i as f64 / (1024.0 * 768.0 / 100.0)) as f64));
        }
        for j in 0..spp {
            let x = i % w;
            let y = h - i / w;
            let mut ray = Ray{o:Vec3{x:0.0, y:0.0, z:0.0}, d:Vec3{x:0.0, y: 0.0, z:0.0}};
            //ray.o = Vec3{x:2.0 * (x as f32) / (w as f32) - 1.0, y:2.0 * (y as f32) / (h as f32) - 1.0, z:5.0};
            //ray.d = Vec3{x:0.0, y:0.0, z:-1.0};
            ray.o = eye;
            ray.d = {
                let tf = (fov*0.5).tan();
                let rpx = 2.0 * (x as f64 + rng.gen_range(0.0, 1.0)) / (w as f64) - 1.0;
                let rpy = 2.0 * (y as f64 + rng.gen_range(0.0, 1.0)) / (h as f64) - 1.0;
                let w = normalize(Vec3{x: aspect * tf * rpx, y: tf * rpy, z:-1.0});
                u_e.mulc(w.x) + v_e.mulc(w.y) + w_e.mulc(w.z)
            };

            let mut l = Vec3{x:0.0, y:0.0, z:0.0};
            let mut th = Vec3{x:1.0, y:1.0, z:1.0};
            for _depth in 0..7 {
                let h = scene.clone().intersect(&ray, &mut 1.0e-4, &mut 1.0e+10);
                if h.is_none() { break; }
                l = l + th * h.unwrap().object.illu;
                ray.o = h.unwrap().p;
                ray.d = { 
                    match h.unwrap().object.mat {
                        Material::Diffuse => {
                            let n = if dot(h.unwrap().n, -ray.d) > 0.0 { h.unwrap().n } else { -h.unwrap().n };
                            let (u, v) = tangent_space(&n);
                            let d = {
                                let r = (rng.gen_range(0.0, 1.0) as f64).sqrt();
                                let t = 2.0 * pi * rng.gen_range(0.0, 1.0);
                                let xr = r * t.cos();
                                let yr = r * t.sin();
                                Vec3{x: xr, y: yr, z: (1.0 - xr * xr - yr * yr).max(0.0).sqrt()}
                            };
                            let ret = u.mulc(d.x) + v.mulc(d.y) + n.mulc(d.z);
                            /*match h.unwrap().object.tp {
                                ObjectType::Triangle => {
                                    print!("{} {} {}\n", ret.x, ret.y, ret.z);
                                }
                                _ => {print!("");}
                            }*/
                            ret
                        }
                        Material::Mirror => {
                            let wi = -ray.d;
                            h.unwrap().n.mulc(2.0 * dot(wi, h.unwrap().n)) - wi
                        }
                        Material::Fresnel => {
                            let wi = -ray.d;
                            let into = dot(wi, h.unwrap().n) > 0.0;
                            let n = if into { h.unwrap().n } else { -h.unwrap().n };
                            let ior = h.unwrap().object.ior;
                            let eta = if into { 1.0 / ior } else { ior };
                            let wt: Option<Vec3> = {
                                let t = dot(wi, n);
                                let t2 = 1.0 - eta * eta * (1.0 - t * t);
                                if t2 < 0.0 { None } else {
                                    Some((n.mulc(t) - wi).mulc(eta) - n.mulc(t2.sqrt()))
                                }
                            };
                            if wt.is_none() {
                                h.unwrap().n.mulc(2.0 * dot(wi, h.unwrap().n)) - wi
                            } else {
                                let fr = {
                                    let cos = if into { dot(wi, h.unwrap().n) } else { dot(wt.unwrap(), h.unwrap().n) };
                                    let r = (1.0 - ior) / (1.0 + ior);
                                    r * r + (1.0 - r * r) * (1.0 - cos).powf(5.0)
                                };
                                if rng.gen_range(0.0, 1.0) < fr {
                                    h.unwrap().n.mulc(2.0 * dot(wi, h.unwrap().n)) - wi
                                } else {
                                    wt.unwrap()
                                }
                            }
                        }
                    }
                };
                th = th * h.unwrap().object.refl;
                if (th.x).max(th.y).max(th.z) < 1.0e-6 {
                    break;
                }
            }
            if j == 0 {
                img.push(l.divc(spp as f64));
            } else {
                let top = img.pop().unwrap();
                img.push(top + l.divc(spp as f64));
                //img.push(Vec3{x: top.x + l.divc(spp as f32), y: top.y + l.divc(spp as f32), z: top.z + l.divc(spp as f32)});
            }
        }
        /*if h.is_some() {
            //let n = h.unwrap().n;
            img.push(h.unwrap().sphere.refl.mulc(dot(&h.unwrap().n, &-ray.d)));
        } else {
            img.push(Vec3{x:0.0, y:0.0, z:0.0});
        }*/
        //});
        //let _ = t.join();
    //}
        //));
    }
    print!("writing...\n");
    for i in img {
        let r = tonemap(i.x);
        let g = tonemap(i.y);
        let b = tonemap(i.z);
        let mut color: String = "".to_string();
        color = color + &r.to_string() + &" ".to_string() + &g.to_string() + &" ".to_string()
            + &b.to_string() + &"\n".to_string();
        //print!("{}", color);
        f.write(color.as_bytes()).unwrap();
    }

}
