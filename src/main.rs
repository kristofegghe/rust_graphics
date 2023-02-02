use hello_world::run;
/* hihi  */

fn _printloop<T: std::fmt::Display>(list:Vec<T>)
{
    for i in list{
        println!("{}",i);
    }
}

    
fn main() {
    // let mut a: Vec<u32>=[1,2,3].to_vec();
    // let inst:Vec<_>=(0..3).flat_map(|y|(0..3).flat_map(move |z| {(0..3).map(move |x| {println!("z {}, y{}, ks{}",z,y,x)}) })).collect();
    // let y=a.iter().map(move | z| {println!("{}",z*2)});
    // _printloop(y);
    // _printloop(a);
    // let instances = (0..4).flat_map(|z| {(0..4).map(move |x| {println!("{},{}",x,z)})});
    // let mut a: Vec<u32>=vec![1,2,3];
    // _printloop(a);
    // _printloop(a);

    pollster::block_on(run());
}

