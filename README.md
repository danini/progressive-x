# Progressive-X

The Progressive-X algorithm proposed in paper: Daniel Barath and Jiri Matas; Progressive-X: Efficient, Anytime, Multi-Model Fitting Algorithm, International Conference on Computer Vision, 2019. 
It is available at https://arxiv.org/pdf/1906.02290

# Installation

1. Getting the sources (and the submodules):
```shell
$ git clone --recursive https://github.com/danini/progressive-x.git
```
or
```shell
$ git clone https://github.com/danini/progressive-x.git
$ cd progressive-x
$ git submodule init
$ git submodule update
```

2. Make a directory for the build files to be generated.
```shell
$ mkdir build_dir
$ cd build_dir
```

3. Configure CMAKE.
```shell
$ cmake-gui ..
```

# Example project

To build the sample project showing examples of fundamental matrix, homography and essential matrix fitting, set variable `CREATE_SAMPLE_PROJECT = ON` when creating the project in CMAKE. 

Next to the executable, copy the `data` folder and, also, create a `results` folder. 

# Requirements

- Eigen 3.0 or higher
- CMake 2.8.12 or higher
- OpenCV 3.0 or higher
- GFlags
- GLog
- A modern compiler with C++17 support


# Acknowledgements

When using the algorithm, please cite `Barath, Daniel, and Matas, Jiří. "Progressive-X: Efficient, Anytime, Multi-Model Fitting Algorithm". Proceedings of the IEEE International Conference on Computer Vision. 2019`.

If you use Progressive-X with Graph-Cut RANSAC as a proposal engine, please cite `Barath, Daniel, and Matas, Jiří. "Graph-cut RANSAC." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018`.
