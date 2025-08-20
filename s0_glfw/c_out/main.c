#include "main.h"

#include <GLFW/glfw3.h>

#include "glfw_h.h"

int main()
{
    GLFWwindow *window = glfw_init(1000, 900);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
    }

    return 0;
}
