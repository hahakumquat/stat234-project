import gym

class CartPoleCroppedGame():

    def __init__(self):
        self.file_prefix = 'cartpole_'
        self.env = gym.make('CartPole-v0').unwrapped
        self.screen_width = 600

    def modify_screen(self, screen):
        view_width = 160
        cart_location = self.get_cart_location()
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (self.screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
        # Strip off the top and bottom of the screen
        return screen[:, 160:320, slice_range]

    def get_cart_location(self):
        world_width = self.env.x_threshold * 2
        scale = self.screen_width / world_width
        return int(self.env.state[0] * scale + self.screen_width / 2.0)
    
