use std::{collections::VecDeque, time::{Duration, Instant}};

pub struct Fps {
    last_time: Option<Instant>,
    frame_time_history: VecDeque<Duration>,
    sum_frame_time: Duration,
    window: u32,
}

impl Fps {
    pub fn new(window: u32) -> Fps {
        Fps {
            last_time: None,
            sum_frame_time: Duration::ZERO,
            frame_time_history: VecDeque::with_capacity(window.try_into().unwrap()),
            window,
        }
    }

    pub fn update(&mut self, time: Instant) {
        if let Some(last_time) = self.last_time {
            let frame_time = time - last_time;
            self.sum_frame_time += frame_time;
            if self.frame_time_history.len() > self.window as usize {
                let popped = self.frame_time_history.pop_back().unwrap();
                self.sum_frame_time -= popped;
            }
            self.frame_time_history.push_front(frame_time);
        }
        self.last_time = Some(time);
    }

    pub fn get_fps(&self) -> f64 {
        if self.frame_time_history.is_empty() {
            return 0.;
        }
        let frame_time = self.sum_frame_time.as_secs_f64() / self.frame_time_history.len() as f64;
        1. / frame_time
    }
}
