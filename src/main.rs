use hello_world::run;

fn main() {
    pollster::block_on(run());
}

