#ifndef MODEL_H
#define MODEL_H

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "shader.h"
#include "mesh.h"
#include "texture.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

class Model{
public:
	Model(std::string path);

	void draw(Shader const &shader) const;
	void setTexture(Texture const &texture);

private:
	void processNode(aiNode *node);
	Mesh processMesh(aiMesh *mesh);
	std::vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, std::string typeName);

	aiScene const *scene;
	std::string const directory;
	std::vector<Mesh> meshes;
	std::unordered_map<std::string, Texture> loadedTexture;

};

#endif