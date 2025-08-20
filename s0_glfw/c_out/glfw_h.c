#include "glfw_h.h"

#include <GLFW/glfw3.h>

GLFWwindow *glfw_init(int window_w, int window_h)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    GLFWwindow *window = glfwCreateWindow(window_w, window_h, "s0_glfw", NULL, NULL);
    glfwMakeContextCurrent(window);

    // trace("OpenGL Vendor: %s", glGetString(GL_VENDOR));
    // trace("OpenGL Renderer: %s", glGetString(GL_RENDERER));
    // trace("OpenGL Version: %s", glGetString(GL_VERSION));

    return window;
}
