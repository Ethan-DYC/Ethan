#!/usr/bin/env python3

import os
import numpy
import random
import string
import cv2
import argparse
import captcha.image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int, default=192)
    parser.add_argument('--height', help='Height of captcha image', type=int, default=96)
    parser.add_argument('--count', help='How many captchas to generate', type=int, default=10000)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str, default='train')
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str, default='symbols.txt')
    parser.add_argument('--font', help='Path to the font file to use in the captcha', type=str, required=False,default='WildCrazy.ttf')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    if args.font:
        # Use the specified font if provided
        captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height, fonts=[args.font])
    else:
        # Default font if no font is specified
        captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height)

    # Read symbols from the specified file
    with open(args.symbols, 'r') as symbols_file:
        captcha_symbols = symbols_file.readline().strip()

    print(f"Generating captchas with symbol set {captcha_symbols} and font {args.font or 'default'}")

    for i in range(args.count):
        # Randomly choose a length between 2 and 6 for the CAPTCHA
        captcha_length = random.randint(1, 6)
        random_str = ''.join([random.choice(captcha_symbols) for _ in range(captcha_length)])

        # Ensure the CAPTCHA does not start with '\'
        while random_str.startswith('\\'):
            random_str = ''.join([random.choice(captcha_symbols) for _ in range(captcha_length)])

        # Construct the image path
        image_path = os.path.join(args.output_dir, random_str + '.png')

        # Check if file already exists and handle filename conflicts
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, f"{random_str}_{version}.png")):
                version += 1
            image_path = os.path.join(args.output_dir, f"{random_str}_{version}.png")

        # Generate and save CAPTCHA image
        image = numpy.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)


if __name__ == '__main__':
    main()