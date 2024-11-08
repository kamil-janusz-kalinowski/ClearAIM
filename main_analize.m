clc; clear; close all;

addpath(genpath("./"));

path_images = "./Materials/1_2mm_brain";
path_masks = "./Results/1_2mm_brain";

paths_pairs = findImageMaskPairs(path_images, path_masks);
paths_pairs = sort_image_mask_struct(paths_pairs);

displayImagesWithMasks(paths_pairs)
%% Calculate and Display Sample Measurements

RADIUS_DILAT = 20;

metrics_array = calcMetricsOfAllImages(paths_pairs, RADIUS_DILAT);
times_days_array = getTimeFromMetadatas([paths_pairs.image_path]);
displayMetrics(metrics_array, times_days_array)

%% Making animation from data
filename_save = 'animation.avi';
createAnimationOfObjectDetection(filename_save, paths_pairs, metrics_array, times_days_array)


function file_pairs = findImageMaskPairs(folder_images, folder_masks, mask_suffix)
    % Retrieves pairs of image and corresponding mask files from specified folders
    %
    % Parameters:
    %    folder_images (string): Path to the folder containing image files
    %    folder_masks (string): Path to the folder containing mask files
    %    mask_suffix (string): Suffix to identify mask files (e.g., '_mask')
    %
    % Returns:
    %    file_pairs (struct): Structure containing pairs of image and mask paths
    %
    % Example:
    %    file_pairs = findImageMaskPairs('images_folder', 'masks_folder', '_mask');
    
    % Set default mask suffix if not provided
    if nargin < 3
        mask_suffix = '_mask';    % Default mask suffix
    end
    
    % List of possible image file extensions
    image_extensions = {'*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.gif'};
    
    % Initialize a structure to store results
    file_pairs = struct('image_path', {}, 'mask_path', {});
    
    % Initialize a list to store image files
    image_files = [];
    
    % Search for images with all extensions
    for ext = image_extensions
        image_files = [image_files; dir(fullfile(folder_images, ext{1}))];
    end
    
    % Iterate over all image files
    for i = 1:length(image_files)
        % Get the image filename without extension
        [~, filename, ext] = fileparts(image_files(i).name);
        
        % Construct the corresponding mask file path in the mask folder
        mask_filename = [filename, mask_suffix, ext];  % Keep the same extension for the mask
        mask_path = fullfile(folder_masks, mask_filename);
        
        % Check if the mask file exists
        if isfile(mask_path)
            % Add the image and mask paths to the structure
            file_pairs(end+1).image_path = fullfile(folder_images, image_files(i).name);
            file_pairs(end).mask_path = mask_path;
        end
    end
end

function binary_mask = processMask(mask, target_size)
    %PROCESSMASK Processes the input mask by converting it to grayscale,
    % resizing it to the target size, and binarizing it.
    %
    % binary_mask = PROCESSMASK(mask, target_size) takes the input mask and
    % processes it by converting it to grayscale (if needed), resizing it to the
    % specified target_size, and binarizing the result.
    %
    % Inputs:
    %   mask - The input mask image.
    %   target_size - The desired size for the output mask [rows, cols].
    %
    % Outputs:
    %   binary_mask - The processed binary mask.

    % If the mask is color, convert it to grayscale
    if size(mask, 3) == 3
        mask = rgb2gray(mask);
    end
    
    % Resize the mask to the target size
    resized_mask = imresize(mask, target_size);
    
    % Binarize the mask
    binary_mask = imbinarize(resized_mask);
end

function displayImageWithMaskContour(image_path, mask_path)
    % displayImageWithMaskContour Displays an image with overlaid mask contours
    %
    % This function loads an original image and a corresponding mask, processes
    % the mask to create a binary version, finds the contours of the mask, and
    % displays the original image with the mask contours overlaid in red.
    %
    % Parameters:
    %    image_path (string): Path to the image file
    %    mask_path (string): Path to the mask file
    %
    % Example:
    %    displayImageWithMaskContour('image.jpg', 'mask.png');
    
    % Load the original image
    original_image = imread(image_path);
    
    % Load the mask
    mask = imread(mask_path);
    
    % Process the mask (using a helper function)
    binary_mask = processMask(mask, [size(original_image, 1), size(original_image, 2)]);
    
    % Find the mask contours
    mask_contours = bwperim(binary_mask);
    
    % Display the original image
    imshow(original_image);
    hold on;
    
    % Overlay the contours on the image (red contour color)
    visboundaries(mask_contours, 'Color', 'r');
    
    hold off;
end

function C = weberContrast(I_sample, I_background)
    %WEBER_CONTRAST Calculates the Weber contrast between a sample and the background
    %
    % C = WEBER_CONTRAST(I_sample, I_background) computes the Weber contrast (C) for a given
    % sample intensity (I_sample) and background intensity (I_background). 
    %
    % The Weber contrast is defined as:
    % C = (I_sample - I_background) / I_background
    %
    % This measure is useful when the background intensity is uniform and the sample's
    % intensity varies slightly from the background.
    %
    % Inputs:
    %   I_sample - Intensity of the sample (scalar or array).
    %   I_background - Intensity of the background (scalar or array of the same size as I_sample).
    %
    % Outputs:
    %   C - The computed Weber contrast (scalar or array).
    %
    % Example:
    %   I_sample = 150;
    %   I_background = 100;
    %   C = weber_contrast(I_sample, I_background);
    %   disp(C);  % Outputs: 0.5
    
    C = (I_sample - I_background) ./ I_background;
end

function sortedStruct = sort_image_mask_struct(inputStruct)
    %SORT_IMAGE_MASK_STRUCT Sorts a structure array based on the numerical values in the image_path field.
    %
    % sortedStruct = SORT_IMAGE_MASK_STRUCT(inputStruct) takes a structure array with fields
    % image_path and mask_path and sorts it according to the numerical values extracted from
    % the image_path field.
    %
    % Inputs:
    %   inputStruct - A structure array with fields 'image_path' and 'mask_path'.
    %
    % Outputs:
    %   sortedStruct - A structure array sorted based on the numerical values in the image_path.

    % Extract image paths
    imagePaths = {inputStruct.image_path};
    
    % Initialize an array to hold numerical values
    nums = zeros(size(imagePaths));

    % Loop through each image path to extract numerical values
    for i = 1:length(imagePaths)
        % Use regexp to find numeric part
        match = regexp(imagePaths{i}, '_(\d+)\.png$', 'tokens'); % Extract numeric part
        if ~isempty(match)
            nums(i) = str2double(match{1}{1}); % Convert to double
        else
            nums(i) = NaN; % Handle cases with no match
        end
    end

    % Sort structure based on the extracted numbers, ignoring NaNs
    [~, sortIdx] = sort(nums); % Sort indices
    sortedStruct = inputStruct(sortIdx);  % Reorder structure based on sorted indices
end


function displayImagesWithMasks(paths_pairs)
    for ind = 1 : length(paths_pairs)
        path_img = paths_pairs(ind).image_path;
        path_mask = paths_pairs(ind).mask_path;
        displayImageWithMaskContour(path_img, path_mask)
    
        progress = (ind / length(paths_pairs)) * 100;
        title(['Processing... ', num2str(progress, '%.1f'), '%']);
    
        pause(0.01)
    end
end

function metrics = calcMetricsFromImageAndMask(img_origin, img_mask, radius_dilat)
    mean_value_object = mean(img_origin(img_mask(:)));
    mask_background = bwmorph(img_mask, 'dilate', radius_dilat) & ~img_mask;
    mean_value_background = mean(img_origin(mask_background(:)));
    
    metrics = initializeStructureForMetrics(1);
    metrics.web_contrast = -weberContrast(mean_value_object, mean_value_background);
    metrics.area = sum(img_mask(:));
    metrics.transmittance = mean_value_object/mean_value_background;
end

function struct_metrics = initializeStructureForMetrics(num_of_elements)
    struct_metrics(num_of_elements) = struct('web_contrast', [], 'area', [], 'transmittance', []);
end

function metrics_array = calcMetricsOfAllImages(paths_pairs, radius_dilate)

    metrics_array = initializeStructureForMetrics(length(paths_pairs));

    for ind = 1 : length(paths_pairs)
        img_origin = imread(paths_pairs(ind).image_path);
        img_mask = imread(paths_pairs(ind).mask_path);
        img_mask = processMask(img_mask, size(img_origin));
    
        metrics_array(ind) = calcMetricsFromImageAndMask(img_origin, img_mask, radius_dilate);
    
        disp("Progress: " + num2str(ind/length(paths_pairs) * 100) + "%")
    end
end

function displayMetrics(metrics_array, time_x)
    area_array = [metrics_array(:).area];
    web_contrast_array = [metrics_array(:).web_contrast];
    transmittance_array = [metrics_array(:).transmittance];

    % Processing data
    area_array = area_array/max(area_array);
    transmittance_array = transmittance_array * 100;
 
    time_x = time_x * 24; % Days to hours

    % Display data for Area
    figure
    plot(time_x, area_array, 'LineWidth', 2)
    title('Area plot')
    xlabel('Time (hours)')  % Added axis label
    ylabel('Normalized Area')  % Added axis label
    
    % Display data for Weber contrast
    figure
    plot(time_x, web_contrast_array, 'LineWidth', 2)
    title('Weber contrast plot')
    xlabel('Time (hours)')  % Added axis label
    ylabel('Weber Contrast')  % Added axis label
    
    % Display data for Transmittance
    figure
    plot(time_x, transmittance_array, 'LineWidth', 2)
    title('Transmittance plot')
    xlabel('Time (hours)')  % Added axis label
    ylabel('Transmittance (%)')  % Added axis label
end


function time = getImageTime(filename)
    % Funkcja zwraca czas wykonania zdjęcia, jeśli jest dostępny w metadanych EXIF.
    % Jeśli nie ma tej informacji, zwraca pustą tablicę.
    
    % Pobierz metadane obrazu
    info = imfinfo(filename);
    
    % Inicjalizuj zmienną time jako pustą
    time = [];
    
    % Sprawdź, czy istnieje informacja o czasie wykonania w metadanych EXIF
    if isfield(info, 'DigitalCamera') && isfield(info.DigitalCamera, 'DateTimeOriginal')
        % Pobierz czas wykonania zdjęcia
        time = info.DigitalCamera.DateTimeOriginal;
    elseif isfield(info, 'FileModDate')
        % Alternatywnie, użyj czasu modyfikacji pliku
        time = info.FileModDate;
    else
        fprintf('Metadane czasowe nie są dostępne dla pliku: %s\n', filename);
    end
end

function numericTime = getImageTimeAsNumeric(filename)
    % getImageTimeAsNumeric Extracts and converts image capture time to numeric format.
    %
    % Description:
    %   This function retrieves the capture time from an image file's metadata 
    %   (EXIF or other metadata fields), and converts it to a numeric format 
    %   representing the number of days since January 0, 0000 in MATLAB's datenum format.
    %   This numeric format allows for easy time-based calculations, such as 
    %   determining time differences between images.
    %
    %   Note: The returned value is in days. To calculate the time difference in 
    %   seconds between two time points, subtract one numeric time from another, 
    %   then multiply by 24 * 3600 (the number of seconds in a day).
    %
    % Input:
    %   filename - A string specifying the path and name of the image file.
    %
    % Output:
    %   numericTime - A numeric value representing the capture time in days since 
    %                 January 0, 0000. Returns an error if no time information is found 
    %                 in the metadata.
    %
    % Example:
    %   timeNum = getImageTimeAsNumeric('image1.jpg');
    %
    %   % To calculate the difference in seconds between two images:
    %   time1 = getImageTimeAsNumeric('image1.jpg');
    %   time2 = getImageTimeAsNumeric('image2.jpg');
    %   timeDifferenceInSeconds = (time2 - time1) * 24 * 3600;
    %
    % Note:
    %   This function requires that the `getImageTime` function is available to extract 
    %   the time as a text string from the image metadata.

    timeStr = getImageTime(filename);
    
    if isempty(timeStr)
        error('Failed to find time information in the metadata.');
    end

    dateTimeObj = datetime(timeStr, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss', 'Locale', 'pl_PL');
    numericTime = datenum(dateTimeObj);
end

function time_array = getTimeFromMetadatas(paths_image)
    time_array = zeros(length(paths_image), 1);
    time_start = getImageTimeAsNumeric(paths_image(1));
    for ind = 1 : length(paths_image)
        time_diff = getImageTimeAsNumeric(paths_image(ind)) - time_start;
        time_array(ind) = time_diff;
    end
end

function createAnimationOfObjectDetection(filename_save, paths_pairs, metrics_array, times_days_array)
    % Create video writer
    videoWriter = VideoWriter(filename_save, 'MPEG-4'); % Name of the output video file
    open(videoWriter);
    

    area_array = [metrics_array(:).area];
    web_contrast_array = [metrics_array(:).web_contrast];
    transmittance_array = [metrics_array(:).transmittance];

    % Processing data
    area_array = area_array/max(area_array);
    transmittance_array = transmittance_array * 100;
 
    times_hours = times_days_array * 24; % Days to hours

    % Create figure for animation
    figure
    % Show mask on image and calculate measurements
    for ind = 1 : length(paths_pairs)
        path_img = paths_pairs(ind).image_path;
        path_mask = paths_pairs(ind).mask_path;
        
        % Display the image with mask contours
        subplot(2, 2, 1);
        displayImageWithMaskContour(path_img, path_mask);
        title('Image with Mask Contours');
        
        % Update progress title
        progress = (ind / length(paths_pairs)) * 100;
        title(['Processing... ', num2str(progress, '%.1f'), '%']);
    
        % Plot area and contrast
        subplot(2, 2, 2);
        plot(times_hours(1:ind), area_array(1:ind), 'LineWidth', 2);
        title('Area Plot');
        xlabel('Time (hours)');
        ylabel('Area normed');
        xlim([times_hours(1), times_hours(end)]);
        ylim([min(area_array), max(area_array)]); % Adjust y-axis for better visibility
    
        subplot(2, 2, 3);
        plot(times_hours(1:ind), web_contrast_array(1:ind), 'LineWidth', 2);
        title('Weber Contrast Plot');
        xlabel('Time (hours)');
        ylabel('Weber Contrast Value');
        xlim([times_hours(1), times_hours(end)]);
        ylim([min(web_contrast_array) max(web_contrast_array)]); % Adjust y-axis for better visibility
    
        subplot(2, 2, 4)
        plot(times_hours(1:ind), transmittance_array(1:ind), 'LineWidth', 2);
        title('Transmittance Plot');
        xlabel('Time (hours)');
        ylabel('Transmittance [%]');
        xlim([times_hours(1), times_hours(end)]);
        ylim([min(transmittance_array) max(transmittance_array)]); % Adjust y-axis for better visibility
    
    
        % Capture the frame
        frameData = getframe(gcf);
        writeVideo(videoWriter, frameData);
    
        % Pause briefly to visualize the process
        pause(0.01);
    end
    
    % Close the video writer
    close(videoWriter);
    
    disp(['Animation completed and saved as ', filename_save]);


end

