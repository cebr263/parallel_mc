import vtkmodules.all as vtk
import time
import tifffile as tiff
import vtkmodules.vtkIOExport as vtkexport
# import mpi4py as MPI
import numpy as np
import os
import sys
import multiprocessing as mp

def numpy_arr_to_polydata(numpy_array):
    cells, points = numpy_array

    vtk_points = vtk.vtkPoints()

    # Add the points to the VTK points object
    for point in points:
        vtk_points.InsertNextPoint(point)

    # Create a VTK cells object
    vtk_cells = vtk.vtkCellArray()

    # Add the cells to the VTK cells object
    for cell in cells:
        vtk_cells.InsertNextCell(3, cell)  # Assuming triangular cells

    # Create a VTK polydata object
    vtk_surface = vtk.vtkPolyData()

    # Set the points and cells for the polydata
    vtk_surface.SetPoints(vtk_points)
    vtk_surface.SetPolys(vtk_cells)

    return vtk_surface

def process_chunk(chunk_indices, tiff_arr, padding_width = 5):
    ## define start and end slice for chunk
    start_slice, end_slice = chunk_indices
    print(chunk_indices)
    ## define full dimensions from full array
    dimensions = tiff_arr.shape
    start_slice = max(0, start_slice - padding_width)
    end_slice = min(dimensions[2] - 1, end_slice + padding_width)
    
    ## define chunk piece to be processed from whole
    chunk_arr = tiff_arr[0:dimensions[0] - 1, 0:dimensions[1] - 1, start_slice:end_slice]
    dimensions = chunk_arr.shape
    print(dimensions)
    input_data = vtk.vtkImageData()
    depth, height, width = chunk_arr.shape
    input_data.SetDimensions(width, height, depth)
    input_data.SetSpacing(1, 1, 1) ## change to whatever actual voxel size is
    input_data.SetOrigin(start_slice,0,0)

    input_data.GetPointData().SetScalars(vtk.vtkUnsignedCharArray()) ## might have to change to different type
    input_data.GetPointData().GetScalars().SetArray(chunk_arr.ravel(),len(chunk_arr.ravel()),1)
    # print(padding_width)


    # Extract the chunk
    chunk = vtk.vtkExtractVOI()
    chunk.SetInputData(input_data)
    chunk.SetVOI(0, dimensions[2] - 1, 0, dimensions[1] - 1, 0, dimensions[0] - 1)

    # Perform marching cubes on the chunk
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputConnection(chunk.GetOutputPort())
    marching_cubes.SetValue(0, 0.5)  # Adjust the threshold as needed

    ## just for debugging!!!
    # stl_writer = vtk.vtkSTLWriter()
    # chunk_name = f"chunk_vols/chunk_{start_slice}-{end_slice}.stl"
    # stl_writer.SetFileName(os.path.join(chunk_name))
    # stl_writer.SetInputConnection(marching_cubes.GetOutputPort())
    # print('write')
    # stl_writer.Write()

    smoothfilter = vtk.vtkSmoothPolyDataFilter()
    smoothfilter.SetInputConnection(marching_cubes.GetOutputPort())
    smoothfilter.SetNumberOfIterations(3)
    smoothfilter.SetRelaxationFactor(0.5)

    smoothfilter.Update()
    # # create numpy arrays storing each surface file
    marching_cubes.Update()
    surface = vtk.vtkPolyData()
    surface.ShallowCopy(smoothfilter.GetOutput())

    num_points = surface.GetNumberOfPoints()
    points = np.zeros((num_points, 3))

    for i in range(num_points):
        points[i] = surface.GetPoint(i)

    num_cells = surface.GetNumberOfCells()
    cells = np.zeros((num_cells, 3), dtype=np.int32)

    for i in range(num_cells):
        cell = surface.GetCell(i)
        cells[i] = [cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)]

    # print('done')
    return [cells, points]


def main(tiff_arr, output_dir, num_processes, padding):

    input_data = vtk.vtkImageData()
    depth, height, width = tiff_arr.shape
    num_slices = depth
    input_data.SetDimensions(width, height, depth)
    input_data.SetSpacing(1, 1, 1) ## change to whatever actual voxel size is
    input_data.SetOrigin(0,0,0)

    input_data.GetPointData().SetScalars(vtk.vtkUnsignedCharArray()) ## might have to change to different type
    input_data.GetPointData().GetScalars().SetArray(tiff_arr.ravel(),len(tiff_arr.ravel()),1)
    
    # # Get the dimensions of the loaded volume
    # reader = vtk.vtkTIFFReader()
    # reader.SetFileName(input_file)
    # reader.Update()
    # dimensions = reader.GetOutput().GetDimensions()
    # num_slices = dimensions[2]

    # Split the volume into chunks based on the number of processes
    chunk_size = num_slices // num_processes
    chunk_indices = [(i * chunk_size, (i + 1) * chunk_size if i < num_processes - 1 else num_slices) for i in range(num_processes)]


    # Create a pool of worker processes to process the chunks in parallel
    start = time.time()

    print('Beginning to pool')
    # with mp.Pool(num_processes) as pool:
    #     # Use apply_async to run the processing function in parallel
    #     process_surfaces = [pool.apply_async(process_chunk, args=(process_id, indices, tiff_arr,padding)) for process_id in range(num_processes)]
    process_surfaces = [None] * len(chunk_indices)
    with mp.Pool(num_processes) as pool:
        for i, indices in enumerate(chunk_indices):
            # process_surfaces = [pool.apply_async(process_chunk, (_, indices, tiff_arr, padding)) for _ in range(num_processes)]
            process_surfaces[i] = pool.apply_async(process_chunk, (indices, tiff_arr, padding))
        # Wait for all processes to complete
        pool.close()
        pool.join()
    end = time.time()
    print('Pooling time:', end - start)


    # Initialize VTK structures
    combined_surface = vtk.vtkPolyData()
    append_filter = vtk.vtkAppendPolyData()

    # Process results as they become available
    for surface in process_surfaces:
        surface_array = surface.get()
        process_surface = numpy_arr_to_polydata(surface_array)
        append_filter.AddInputData(process_surface)
    append_filter.Update()
    combined_surface.DeepCopy(append_filter.GetOutput())



    stl_writer = vtk.vtkSTLWriter()
    chunk_name = f"chunk_0-0.stl"
    stl_writer.SetFileName(os.path.join(output_dir, chunk_name))
    stl_writer.SetInputData(combined_surface)
    stl_writer.SetFileTypeToASCII()
    print("Beginning to write")
    start = time.time()
    stl_writer.Write()
    end = time.time()
    

    # Open and read the STL file to ensure printing correctly
    stl_file_path = os.path.join(output_dir, chunk_name)
    print(stl_file_path) 
    try:
        print('Write time:',end - start)
        with open(stl_file_path, "r") as stl_file:
            # Read and print the first 5 lines
            for i, line in enumerate(stl_file):
                print(line.strip())
                if i >= 4:  # Print the first 5 lines (0 to 4)
                    break
    except FileNotFoundError:
        print(f"STL file '{stl_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    ## turn tiff into numpy array
    tiff_arr = tiff.imread('reslice_matrix_bin4x_min.tif')
    if len(tiff_arr.shape) < 3:
        print('Error: Recieved TIFF image not volume!')
        sys.exit()
    tiff_arr = np.pad(tiff_arr, pad_width=2, mode='constant', constant_values=0)
    tiff_arr = tiff_arr.astype(np.uint8)
    print(tiff_arr.shape)
    print(tiff_arr.dtype)

    # input_file = "203vox_nohole.tif"
    output_dir = "chunk_vols"
    num_processes = 10  # Adjust the number of processes as needed
    padding_width = 1
    # Iterate over all files in the folder and delete them
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            elif os.path.isdir(file_path):
                # If you want to delete files in subdirectories as well, you can use os.rmdir to delete directories
                # Note: Be cautious when deleting directories; make sure you really want to delete them.
                os.rmdir(file_path)
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(tiff_arr, output_dir, num_processes, padding_width)
