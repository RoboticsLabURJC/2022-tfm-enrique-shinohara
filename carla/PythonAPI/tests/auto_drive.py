
import pygame

def init_screen_and_clock():
    global screen, display, clock
    pygame.init()
    WINDOW_SIZE = (1150, 640)
    pygame.display.set_caption('Game')
    screen = pygame.display.set_mode(WINDOW_SIZE, 0, 32)
    clock = pygame.time.Clock()


def create_fonts(font_sizes_list):
    "Creates different fonts with one list"
    fonts = []
    for size in font_sizes_list:
        fonts.append(
            pygame.font.SysFont("Arial", size))
    return fonts


def render(fnt, what, color, where):
    "Renders the fonts as passed from display_fps"
    text_to_show = fnt.render(what, 0, pygame.Color(color))
    screen.blit(text_to_show, where)


def display_fps():
    "Data that will be rendered and blitted in _display"
    render(
        fonts[0],
        what=str(int(clock.get_fps())),
        color="white",
        where=(0, 0))


init_screen_and_clock()
# This create different font size in one line
fonts = create_fonts([32, 16, 14, 8])

loop = 1
while loop:
    screen.fill((0, 0, 0))
    display_fps()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            loop = 0
    clock.tick(60)
    pygame.display.flip()

pygame.quit()
print("Game over")