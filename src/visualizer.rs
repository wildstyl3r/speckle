use image::RgbaImage;
use std::num::NonZeroU32;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

//this boilerplate was intentionally adapted from AI generation

struct ImageVisualizerApp {
    image: RgbaImage,
    window: Option<Arc<Window>>,
    softbuffer_context: Option<softbuffer::Context<Arc<Window>>>,
    softbuffer_surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
}

impl ApplicationHandler for ImageVisualizerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("preview")
                .with_inner_size(winit::dpi::LogicalSize::new(
                    self.image.width() * 10,
                    self.image.height() * 10,
                ))
                .with_transparent(true);

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            let context = softbuffer::Context::new(window.clone()).unwrap();
            let mut surface = softbuffer::Surface::new(&context, window.clone()).unwrap();

            let size = window.inner_size();
            surface
                .configure(
                    NonZeroU32::new(size.width).unwrap(),
                    NonZeroU32::new(size.height).unwrap(),
                    softbuffer::AlphaMode::Premultiplied,
                )
                .unwrap();

            self.window = Some(window);
            self.softbuffer_context = Some(context);
            self.softbuffer_surface = Some(surface);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let (Some(window), Some(surface)) = (&self.window, &mut self.softbuffer_surface)
                {
                    let size = window.inner_size();
                    if let (Some(w), Some(h)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                    {
                        surface.resize(w, h).unwrap();
                        let mut buffer = surface.next_buffer().unwrap();
                        let pixels = buffer.pixels();

                        let img_width = self.image.width();
                        let img_height = self.image.height();

                        for y in 0..size.height {
                            for x in 0..size.width {
                                let src_x = x * img_width / size.width;
                                let src_y = y * img_height / size.height;

                                let pixel = self.image.get_pixel(src_x, src_y);

                                let buffer_index = (y * size.width + x) as usize;

                                pixels[buffer_index] = softbuffer::Pixel::new_rgba(
                                    pixel[0], pixel[1], pixel[2], pixel[3],
                                );
                            }
                        }
                        buffer.present().unwrap();
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. }
                if event.state.is_pressed()
                    && event.logical_key
                        == winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape) =>
            {
                event_loop.exit();
            }
            _ => {}
        }
    }
}

pub fn display_image_buffer(image: RgbaImage) -> Result<(), EventLoopError> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);

    let mut app = ImageVisualizerApp {
        image,
        window: None,
        softbuffer_context: None,
        softbuffer_surface: None,
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}
