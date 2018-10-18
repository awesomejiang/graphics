#include "framebuffer.h"

Framebuffer::Framebuffer(int width, int height, int colors)
: width(width), height(height), colorBuffers(colors){
	//init framebuffer object
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	//generate texture
	glGenTextures(colors, colorBuffers.data());
	for(auto i=0; i<colors; ++i){
		glBindTexture(GL_TEXTURE_2D, colorBuffers[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		//glBindTexture(GL_TEXTURE_2D, 0);
 
		//attach it to framebuffer object
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+i, GL_TEXTURE_2D, colorBuffers[i], 0);
	}
	std::vector<unsigned int> attachments(colors);
	std::iota(attachments.begin(), attachments.end(), GL_COLOR_ATTACHMENT0);
	glDrawBuffers(colors, attachments.data());

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