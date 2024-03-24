print ("test")

color = "5c0d0f"

def hex_to_rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

vla = hex_to_rgb(color) # (255, 165, 1)
print (vla)