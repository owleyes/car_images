import openai
import random
import requests
import os


# Set up the OpenAI client with xAI API (compatible with OpenAI SDK)
# Replace 'your_api_key_here' with your actual xAI API key obtained from https://x.ai/api
client = openai.OpenAI(
    base_url="https://api.x.ai/v1",
    api_key="X_API_KEY"
)

# Define lists for variations
car_brands = ["Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes-Benz", "Volkswagen", "Tesla", "Nissan", "Hyundai"]
car_models = {
    "Toyota": ["Corolla", "Camry", "Prius", "RAV4", "Highlander"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey"],
    "Ford": ["F-150", "Mustang", "Explorer", "Escape", "Focus"],
    "Chevrolet": ["Silverado", "Malibu", "Equinox", "Tahoe", "Camaro"],
    "BMW": ["3 Series", "5 Series", "X3", "X5", "7 Series"],
    "Mercedes-Benz": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
    "Volkswagen": ["Golf", "Passat", "Tiguan", "Atlas", "Jetta"],
    "Tesla": ["Model 3", "Model Y", "Model S", "Model X", "Cybertruck"],
    "Nissan": ["Altima", "Rogue", "Sentra", "Pathfinder", "GT-R"],
    "Hyundai": ["Elantra", "Sonata", "Tucson", "Santa Fe", "Palisade"]
}
colors = ["silver", "black", "white", "red", "blue", "green", "gray", "yellow", "orange", "purple"]
damage_types = [
    "windshield crack",
    "sideswipe dent on the driver's side",
    "rear end collision damage",
    "front bumper smash",
    "t-bone impact on the passenger side",
    "hood crumple from head-on accident",
    "taillight shatter",
    "door panel scrape",
    "fender bend",
    "roof dent from rollover"
]
years = list(range(2015, 2026))

# Create a directory to save images
os.makedirs("generated_images", exist_ok=True)

# Generate 1000 images
for i in range(1000):
    brand = random.choice(car_brands)
    model = random.choice(car_models[brand])
    color = random.choice(colors)
    year = random.choice(years)
    damage = random.choice(damage_types)
    
    prompt = (
        f"photorealistic {year} {brand} {model} in {color} that is in a parking lot "
        f"with vehicle damage from an accident: {damage}. Zoom into the damage. "
        "It should look real from a camera."
    )
    
    # Call the API to generate the image
    # Note: Using 'grok-2-image-1212' as the model since image generation with Grok 4 is coming soon.
    # Check https://docs.x.ai/docs/models for updates on Grok 4 image capabilities.
    response = client.images.generate(
        model="grok-2-image-1212",
        prompt=prompt,
        n=1,  # Generate one image per call
        #size="1024x1024"  # Assuming standard size; adjust if needed based on API docs
    )
    
    # Get the image URL from the response
    image_url = response.data[0].url
    
    # Download and save the image
    image_path = os.path.join("generated_images", f"image_{i+1}.png")
    with open(image_path, "wb") as f:
        f.write(requests.get(image_url).content)
    
    print(f"Generated and saved image {i+1}: {image_path}")

print("Image generation complete.")