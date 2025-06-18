# CUDA AWGN Signal Generator

This project generates a complex signal with additive white Gaussian noise (AWGN) using CUDA. It serves as an example of how to utilize CUDA for signal processing tasks.

## Project Structure

```
cuda-awgn-signal
├── src
│   ├── main.cu        # Entry point of the application
│   ├── signal.cu      # Functions to generate complex signals and add AWGN
│   ├── signal.h       # Header file for signal functions
│   └── utils.cu       # Utility functions for drawing signals
├── CMakeLists.txt     # CMake configuration file
└── README.md           # Project documentation
```

## Requirements

- CUDA Toolkit
- CMake

## Building the Project

1. Clone the repository or download the project files.
2. Open a terminal and navigate to the project directory.
3. Create a build directory:
   ```
   mkdir build
   cd build
   ```
4. Run CMake to configure the project:
   ```
   cmake ..
   ```
5. Build the project:
   ```
   make
   ```

## Running the Application

After building the project, you can run the application with the following command:

```
./cuda-awgn-signal
```

## Functionality

- **Signal Generation**: The application generates a complex signal.
- **AWGN Addition**: Additive white Gaussian noise is added to the generated signal.
- **Signal Visualization**: The signal can be visualized using the utility functions provided.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.