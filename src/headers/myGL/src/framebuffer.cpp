#include "framebuffer.h"

Framebuffer::Framebuffer(int width, int height, std::vector<GLint> texFormats)
: width(width), height(height), colorBuffers(texFormats.size()){
	//init framebuffer object
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	//generate texture
	int sz = colorBuffers.size();
	glGenTextures(sz, colorBuffers.data());
	for(auto i=0; i<sz; ++i){
		glBindTexture(GL_TEXTURE_2D, colorBuffers[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        if(texFormats[i] == GL_RGB)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        else if(texFormats[i] == GL_R32F)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
        else if(texFormats[i] == GL_RGB32F)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
		else{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
			std::runtime_error("warning in framebuffer: this type of texutre is not implemented yet, converted to GL_RGB ");
		}
		//glBindTexture(GL_TEXTURE_2D, 0);
 
		//attach it to framebuffer object
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+i, GL_TEXTURE_2D, colorBuffers[i], 0);
	}
	std::vector<unsigned int> attachments(sz);
	std::iota(attachments.begin(), attachments.end(), GL_COLOR_ATTACHMENT0);
	glDrawBuffers(sz, attachments.data());

	//generate renderbuffer for depth and stencil test
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

	if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
		throw std::runtime_error("Error in Framebuffer::Framebuffer(): Framebuffer is not complete.");

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Framebuffer::bind() const{
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
}

void Framebuffer::unbind() const{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Framebuffer::clearBuffers() const{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

unsigned int Framebuffer::getTex(unsigned int const &idx) const {
	return colorBuffers[idx];
}