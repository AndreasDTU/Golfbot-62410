

#Color gets an id
colors = {
    "Floor" : 0,
    "Ball" : 1,
    "Orange Ball" : 2,
    "Robot" : 3,
    "Wall" : 4
}

#Make 2d array for coordinate system
#Some shit like this idk???
def pictocord(img, x_max, y_max):
    coordinate_array = [[0 for _ in range(x_max)] for _ in range(y_max)] 

    #Fill 2d array with color values
    for y in range(y_max):
        for x in range (x_max):
            coordinate_array[y][x] = colors.get(img.getpixel((x, y)), 0) #This is bullshit and dont work pls fix guys
    return coordinate_array






