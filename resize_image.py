from PIL import Image


path = "/root/CoDeF/all_sequences/heygen/base_control/canonical_0.png"

# Open the original image
original_image = Image.open(path)

# Resize the image
resized_image = original_image.resize((1080, 1080))

# Save the resized image
resized_image.save(path)
