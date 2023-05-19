from random import randint
from PIL import Image, ImageDraw
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import concurrent.futures
import os

# Set the image size and line thickness
WIDTH, HEIGHT = 400, 100
LINE_THICKNESS = 4
num_classes = 50

N = 1000

print(os.getcwd())


def convert_file_num(num):
    return str(int(num)) if num >= 10 else "0" + str(int(num))


# Create a new image and a draw object to draw on it
img = Image.new("L", (WIDTH, HEIGHT), color="white")
draw = ImageDraw.Draw(img)
if not os.path.exists(f"data"):
    os.mkdir(f"data")
if not os.path.exists(f"testdata"):
    os.mkdir(f"testdata")
for i in range(num_classes):
    if not os.path.exists(f"data/{convert_file_num(i)}"):
        os.mkdir(f"data/{convert_file_num(i)}")
for i in range(num_classes):
    if not os.path.exists(f"testdata/{convert_file_num(i)}"):
        os.mkdir(f"testdata/{convert_file_num(i)}")

# Draw each line


@njit
def gen(number_of_lines, number_of_images):
    images = np.zeros(shape=(number_of_images, number_of_lines, 2, 3))
    for n in range(number_of_images):
        pre_x = 0
        num_lines = randint(0, number_of_lines)
        line_spacing = (WIDTH - LINE_THICKNESS) / (num_lines + 1)
        for i in range(num_lines):
            a = np.random.uniform(0.5, 1.5)
            x = a * line_spacing + pre_x
            pre_x = x
            min_distance_from_edge = 30  # you can adjust this value
            if x < min_distance_from_edge:
                x = min_distance_from_edge
            elif x > WIDTH - min_distance_from_edge:
                x = WIDTH - min_distance_from_edge
            cur = np.random.uniform(max(-40, -x), min(40, WIDTH - x))
            height_bot = np.random.uniform(HEIGHT * 0.7, HEIGHT)
            height_top = np.random.uniform(0, 0.3 * HEIGHT)
            images[n, i, 0, 0] = x + cur
            images[n, i, 0, 1] = height_top
            images[n, i, 1, 0] = x - cur
            images[n, i, 1, 1] = height_bot

            images[n, i, 1, 2] = num_lines
    return images


gen(1, 1)
num_threads = 8


def generate_image(data_name, N):
    # Create a thread pool with the specified number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit generator() function to the thread pool and return a list of futures
        futures = [
            executor.submit(lambda: img_gen(N * i, data_name, N))
            for i in range(num_threads)
        ]
        # Wait for all futures to complete and retrieve their results
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    return results


def img_gen(start_index, data_name, N):
    images = gen(num_classes - 1, N)
    for index, i in enumerate(images):
        img = Image.new("L", (WIDTH, HEIGHT), color="white")
        draw = ImageDraw.Draw(img)
        for n in i:
            ran_int = np.random.randint(1, 4)
            draw.line(
                [(n[0][0], n[0][1]), (n[1][0], n[1][1])], fill="black", width=ran_int
            )

        folder_name = convert_file_num(i[0][1][2])
        file_path = f"{data_name}/{folder_name}/{index+start_index}.png"

        img.save(file_path, mode="L")
        # This will print True if the file exists


generate_image("data", N)
generate_image("testdata", int(N * 0.2))


# Show the image (optional)
