import math
import keyboard
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)



NUM_METABALLS = 6
SIZE_METABALLS = 5000
BALLS_SPEED = 6

"""WIN_WIDTH = 800
WIN_HEIGHT = 600"""

WIN_WIDTH = 1536
WIN_HEIGHT = 864

gui = ti.GUI("What is this", (WIN_WIDTH, WIN_HEIGHT), fullscreen=True, background_color=0x25A6D9)

pixels = ti.Vector.field(3, dtype=float, shape=(WIN_WIDTH, WIN_HEIGHT))

@ti.data_oriented
class metaball_system:

    def __init__(self, num_balls: ti.i32):
        self.num_balls = num_balls
        self.points_pos = ti.Vector.field(2, dtype=ti.f32, shape=(num_balls, 1))
        self.points_vel = ti.Vector.field(2, dtype=ti.f32, shape=(num_balls, 1))
        self.points_rad = ti.field(dtype=float, shape=(num_balls, 1))

    @ti.kernel
    def move(self):
        for i in range(self.num_balls):
            if not 0 < self.points_pos[i, 0][0] < WIN_WIDTH:
                self.points_vel[i, 0][0] = -self.points_vel[i, 0][0]
            if not 0 < self.points_pos[i, 0][1] < WIN_HEIGHT:
                self.points_vel[i, 0][1] = -self.points_vel[i, 0][1]
            self.points_pos[i, 0] = self.points_pos[i, 0] + self.points_vel[i, 0]

    @ti.kernel
    def initialize_metaballs(self):
        for x in range(self.num_balls):
            self.points_pos[x, 0] = ti.Vector([ti.random(ti.f32) * WIN_WIDTH, ti.random(ti.f32) * WIN_HEIGHT])
            self.points_vel[x, 0] = ti.Vector([(ti.random(ti.f32) - 0.5) * BALLS_SPEED, (ti.random(ti.f32) - 0.5) * BALLS_SPEED])
            self.points_rad[x, 0] = (ti.random(ti.f32) + 0.5) * SIZE_METABALLS

    @ti.kernel
    def set_pixels(self, time: ti.f32):
        for i, j in pixels:
            intensity = 0.0
            for x in range(self.num_balls):
                d = pow(ti.cast(i, float) - self.points_pos[x, 0][0], 2) + pow(ti.cast(j, float) - self.points_pos[x, 0][1], 2)
                intensity += self.points_rad[x, 0] / d
            #pixels[i, j] = ti.Vector([min(intensity * (ti.sin(time) + 0.5), 200), min(intensity * (ti.sin(time * 1.3) + 0.5), 150), min(intensity * (ti.sin(time * 1.7) + 0.5), 200)])
            pixels[i, j] = ti.Vector(
                [min(intensity * (ti.sin(1.6) + 0.5), 0.7), min(intensity * (ti.sin(1.6 * 1.3) + 0.5), 0.7),
                 min(intensity * (ti.sin(1.6 * 1.7) + 0.5), 0.7)])


if __name__ == '__main__':

    n_balls = NUM_METABALLS
    p_sys = metaball_system(n_balls)
    p_sys.initialize_metaballs()

    running = True
    time = 0.0
    while(running and gui.running):
        time += 0.003
        if keyboard.is_pressed('Esc'):
            running = False
        if keyboard.is_pressed('q'):
            print(time)
        p_sys.move()
        p_sys.set_pixels(time)
        gui.set_image(pixels)
        gui.show()





