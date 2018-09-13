#include "texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//texture from ID
Texture::Texture(unsigned int const & ID, std::string const & type, std::string const &name)
:ID(ID), type(type), name(name) {}

//2d texture
Texture::Texture(std::string const &path, std::string const &type, std::string const &name)
: type(type), name(name){
    glGenTextures(1, &ID);

    //TODO: some texture has to be flipped./some texture are not well aligned. How to determine?
    stbi_set_flip_vertically_on_load(true);
    //glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    int width, height, nrComponents;
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrComponents, 0);
    if (data){
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;
        
        glBindTexture(GL_TEXTURE_2D, ID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else{
        stbi_image_free(data);
    	throw std::runtime_error("Texture failed to load at path: " + path);
    }
    //reset it back to default
    stbi_set_flip_vertically_on_load(false);
}

//cube texture
Texture::Texture(std::vector<std::string> const &paths, std::string const &type, std::string const &name)
: type(type), name(name){
    glGenTextures(1, &ID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, ID);

	int width, height, nrChannels;
	for(unsigned int i = 0; i < paths.size(); i++){
		unsigned char *data = stbi_load(paths[i].c_str(), &width, &height, &nrChannels, 0);
		if (data){
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
			stbi_image_free(data);
		}
		else{
        	stbi_image_free(data);
    		throw std::runtime_error("Texture failed to load at path: " + paths[i]);
		}
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}