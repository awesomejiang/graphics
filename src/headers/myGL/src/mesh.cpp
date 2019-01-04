#include "mesh.h"

Mesh::Mesh(std::vector<Vertex> const &vertices,
std::vector<unsigned int> const &indices,
std::vector<Texture> const &textures)
:pts(indices.size()), textures(textures){
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	//set VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

	//set EBO
	//if input indices is empty, draw points in vertices in order;
	std::vector<unsigned int> indicesInMesh;
	if(indices.empty()){
		pts = vertices.size();
		indicesInMesh.resize(pts);
  		std::iota(indicesInMesh.begin(), indicesInMesh.end(), 0);
  	}
  	else
  		indicesInMesh.assign(indices.begin(), indices.end());

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, pts*sizeof(unsigned int), indicesInMesh.data(), GL_STATIC_DRAW);

    //set vertex postions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    //set normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));

    //set texCoords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));

    //unbind VAO
    glBindVertexArray(0);
}


void Mesh::draw(Shader const &shader, GLenum const &mode) const{
	unsigned int diffIdx = 1, specIdx = 1, normalIdx = 1, heightIdx = 1;
	for(auto i=0; i<textures.size(); ++i){
		glActiveTexture(GL_TEXTURE0 + i);

		auto ID = textures[i].ID;
		std::string type = textures[i].type;
		//texture.name is actually not sued for colormaps
		if(type == "textureDiffuse" || type == "textureSpecular" || type == "textureNormal" || type == "textureHeight"){
			if(type == "textureDiffuse")
				type += std::to_string(diffIdx++);
			else if(type == "textureSpecular")
				type += std::to_string(specIdx++);
			else if(type == "textureNormal")
				type += std::to_string(normalIdx++);
			else if(type == "textureHeight")
				type += std::to_string(heightIdx++);

			shader.setUniform("material." + type, i);
			glBindTexture(GL_TEXTURE_2D, ID);
		}
		else if(type == "texture2D"){
			shader.setUniform(textures[i].name, i);
			glBindTexture(GL_TEXTURE_2D, ID);
		}
		else if(type == "textureCube"){
			shader.setUniform(textures[i].name, i);
			glBindTexture(GL_TEXTURE_CUBE_MAP, ID);
		}
		else{
			throw std::runtime_error("error: unrecognized texture type.");
		}
	}

	//draw
	glBindVertexArray(VAO);
	glDrawElements(mode, pts, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	//set everything to default
	glActiveTexture(GL_TEXTURE0);
}


void Mesh::setTexture(Texture const &texture){
	textures.push_back(texture);
}

unsigned int Mesh::getTexID(int index)const{
	return textures[index].ID;
}

std::string Mesh::getTexName(int index)const{
	return textures[index].name;
}