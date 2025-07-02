#!/usr/bin/env python3
"""
generate_damaged_cars.py

Batch-creates photorealistic images of accident-damaged cars using the OpenAI Images API (DALL·E 3).
"""
import argparse
import asyncio
import os
import sys
import random
import base64
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
try:
    # Requires openai-python >=1.0.0
    from openai import AsyncOpenAI, RateLimitError
except ImportError:
    print(
        "Error: AsyncOpenAI not found. Please upgrade openai package to >=1.0.0.",
        file=sys.stderr
    )
    sys.exit(1)
import pandas as pd
import aiofiles

# Load environment variables from .env file, if present
load_dotenv()

VEHICLES: List[Tuple[int, str, str, str]] = [
    (2010, "Toyota", "Camry", "Red"),
    (2015, "Honda", "Civic", "Blue"),
    (2020, "Tesla", "Model 3", "White"),
    (2018, "Ford", "F-150", "Black"),
    (2012, "BMW", "X5", "Silver"),
    (2019, "Nissan", "Leaf", "Green"),
    (2016, "Chevrolet", "Silverado", "Gray"),
    (2021, "Audi", "Q5", "White"),
    (2017, "Jeep", "Wrangler", "Yellow"),
    (2022, "Volkswagen", "ID.4", "Blue"),
    (2014, "Subaru", "Outback", "Brown"),
    (2013, "Mercedes-Benz", "C-Class", "Black"),
    (2023, "Hummer", "EV SUV", "Orange"),
    (2009, "Dodge", "Charger", "Red"),
    (2011, "Lexus", "RX", "Silver"),
]

DAMAGE_SCENARIOS: List[str] = [
    "front-end crush",
    "rear-end crush",
    "side (T-bone) impact",
    "sideswipe",
    "roof crush / rollover",
    "wheel & suspension failure",
    "glass-only damage",
    "minor fender-bender",
    "hail dents",
    "flood line",
    "engine-bay fire",
    "fallen-tree impact",
    "total loss",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch generate accident-damaged car images using OpenAI DALL·E 3."
    )
    parser.add_argument(
        "--max-images", type=int, default=100,
        help="Maximum number of images to generate (default: 100)"
    )
    parser.add_argument(
        "--openai-key", type=str,
        help="OpenAI API key (fallback to OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./out",
        help="Directory to save images and metadata (default: ./out)"
    )
    return parser.parse_args()


def build_prompt(vehicle: Tuple[int, str, str, str], damage: str) -> str:
    """Construct the image prompt for a given vehicle and damage scenario."""
    year, make, model, color = vehicle
    return (
        f"A photorealistic {year} {make} {model} — {color} — parked on a plain asphalt lot "
        f"under soft overcast daylight at a 45-degree front-quarter angle, aperture f/4. "
        f"Depict **{damage}** in vivid detail; undamaged areas remain factory-fresh. "
        f"No people, no text, no watermarks."
    )


async def generate_image(
    client: AsyncOpenAI,
    vehicle: Tuple[int, str, str, str],
    damage: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    metadata: List[Dict[str, Any]]
) -> None:
    """Generate one image via OpenAI, save it, and record metadata."""
    prompt = build_prompt(vehicle, damage)
    make_slug = vehicle[1].replace(" ", "_").lower()
    model_slug = vehicle[2].replace(" ", "_").lower()
    damage_slug = (
        damage.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )
    filename = f"{make_slug}_{model_slug}_{damage_slug}.jpg"
    output_path = os.path.join(args.output_dir, filename)
    backoff_base = 1.0

    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.images.generate(
                    prompt=prompt,
                    size="1024x1024",
                    n=1,
                    response_format="b64_json"
                )
                # The new ImagesResponse is a Pydantic model: access via attributes
                b64_json = response.data[0].b64_json
                image_bytes = base64.b64decode(b64_json)
                async with aiofiles.open(output_path, "wb") as img_file:
                    await img_file.write(image_bytes)
                timestamp = datetime.now(timezone.utc).isoformat()
                metadata.append({
                    "timestamp": timestamp,
                    "file_path": output_path,
                    "prompt": prompt
                })
                print(f"[{timestamp}] Saved image: {output_path}")
                return
            except RateLimitError:
                wait = backoff_base * (2 ** attempt) + random.random()
                print(f"Rate limit hit, retrying in {wait:.2f}s...", file=sys.stderr)
                await asyncio.sleep(wait)
        print(f"Failed to generate after retries: {prompt}", file=sys.stderr)


async def main(args: argparse.Namespace) -> None:
    """Coordinate the batch image generation tasks and write metadata."""
    key = args.openai_key or os.getenv("OPENAI_API_KEY")
    if not key:
        print(
            "Error: Provide OpenAI API key via --openai-key or OPENAI_API_KEY env var.",
            file=sys.stderr
        )
        sys.exit(1)

    # instantiate the v1 async client now that we have the key
    client = AsyncOpenAI(api_key=key)
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare the list of (vehicle, damage) pairs
    pairs = [(v, d) for v in VEHICLES for d in DAMAGE_SCENARIOS]
    if args.max_images < len(pairs):
        pairs = pairs[: args.max_images]

    semaphore = asyncio.Semaphore(5)
    metadata: List[Dict[str, Any]] = []

    tasks = [
        asyncio.create_task(generate_image(client, v, d, args, semaphore, metadata))
        for v, d in pairs
    ]
    await asyncio.gather(*tasks)

    # Write metadata to CSV
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metadata written to {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))


"""
README:

Setup:
  1. pip install -r requirements.txt
  2. Create a .env file with:
       OPENAI_API_KEY=your_api_key_here

Usage:
  python generate_damaged_cars.py [--max-images N] [--openai-key KEY] [--output-dir PATH]
"""

