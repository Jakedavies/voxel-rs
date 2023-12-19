mod state;



fn main() {
    pollster::block_on(state::run());
}

