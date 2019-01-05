#Jiawei's Graphics Projects

## Final Target
A particle-based simulation system for simulating VFX like fluid/light/shadow/smoke...

## Todo List(keep updating)
1. fluid Surface rendering.
2. optimize the uniform grid cell implementation.
3. shadow rendering.
4. obstacle support.
5. model loading system(based on `renderEngine`)
6. GUI(maybe use `IMGUI`?)
7. Cmake system


## Deperacated works
### hsSimulator
HS Package-open Simulator aiming to practice OpenGL light VFX shader implementations. 

### fluidGrid
Fluid system based on grid algorithm. Works well for "dye advection"-type simulations. However, I change my mind to implement my system using particle-based method.

### renderEngine
Work as a OpenGL wrapper system.    
Load models and render parameters from file. Should support spatial translations ,shadow renderings and deferred light renderings.  
Thinking about give it a GUI.

### particleSystem
My first try for CUDA-OpenGL coop. Functionalities should be integrated into final work.

## 3rd Party Libraries
### 1. GLFW
#### download
    http://www.glfw.org/download.html
#### dependency
    sudo apt-get install xorg-dev
    sudo apt install libgl1-mesa-dev
#### install
    mkdir build
    cd build
    cmake [glfw folder]
    make
    sudo make install

### 2. GLAD
#### download
    http://glad.dav1d.de/
    gl: ver3.3+
    Profile: Core
    Generate a loader: ticked
#### install
    sudo mv glad/ KHR/ /usr/local/include
    sudo mv glad.c [my project folder]

### 3. stb_image.h
#### download
    wget https://github.com/nothings/stb/blob/master/stb_image.h

#### install
    sudo mv stb_image.h [my project folder]

### 4. glm
#### download
    https://glm.g-truc.net/0.9.9/index.html
#### install
    mkdir build
    cd build
    cmake [glm folder]
    make
    sudo make install

### 5. assimp
#### download
    https://github.com/assimp/assimp/releases/tag/v4.1.0
#### install
    mkdir build
    cd build
    cmake [assimp folder]
    make
    sudo make install
