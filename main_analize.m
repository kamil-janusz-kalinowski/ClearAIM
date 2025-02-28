clc; clear; close all;

addpath(genpath("./"));

path_images = "./Materials/1mm brain/skrawek 5_po-recenzjach";
path_masks = "./Results/1mm brain/skrawek 5_po-recenzjach";
filename_save = '1mm brain skrawek 5_po-recenzjach';

paths_pairs = findImageMaskPairs(path_images, path_masks);
paths_pairs = sort_image_mask_struct(paths_pairs);

%paths_pairs = paths_pairs(1:100); %TEMP

%displayImagesWithMasks(paths_pairs)
%% Calculate and Display Sample Measurements
RADIUS_DILAT = 20;
metrics_array = calcMetricsOfAllImages(paths_pairs, RADIUS_DILAT);
%%
range_temp = 150:length(paths_pairs);

% times_days_array = getTimeFromMetadatas([paths_pairs(range_temp).image_path]);
% times_days_array = times_days_array - min(times_days_array);
times_days_array = getTimeFromName([paths_pairs(range_temp).image_path]);

metrics_array_part = metrics_array(range_temp);
paths_pairs_part = paths_pairs(range_temp);

inds = getIndsForTimeSynchronization(times_days_array);
times_days_array = times_days_array(inds); metrics_array_part = metrics_array_part(inds); paths_pairs_part = paths_pairs_part(inds);

% TEMP: Dodatkowe przycięcie by sprawić że każdy film trwa tyle samo czasu
inds = times_days_array < (71/24);
times_days_array = times_days_array(inds);
metrics_array_part = metrics_array_part(inds);
paths_pairs_part = paths_pairs_part(inds);

displayMetrics(metrics_array_part, times_days_array);

save(filename_save + ".mat", 'times_days_array', 'metrics_array_part');

%% Making animation from data
createAnimationOfObjectDetection(filename_save, paths_pairs_part, metrics_array_part, times_days_array)

%% TEMP

clc; clear; close all;

addpath(genpath("./"));

path_images = "./Materials/1mm brain/skrawek 2";
path_masks = "./Results/1mm brain/skrawek 2";
filename_save = 'test';

paths_pairs = findImageMaskPairs(path_images, path_masks);
paths_pairs = sort_image_mask_struct(paths_pairs);

% Get dark and bright sample

ind_dark = 10;
ind_light = length(paths_pairs) - 10;

image_path = paths_pairs(ind_dark).image_path;
mask_path = paths_pairs(ind_dark).mask_path;
original_image = imread(image_path);

resized_image = imresize(original_image, 1/5);
% Load the mask
mask = imread(mask_path);
% Process the mask (using a helper function)
mask_SAM = imresize(mask, size(original_image), 'nearest');

% Oblicz próg metodą Otsu
threshold = graythresh(resized_image);
% Zastosuj próg do segmentacji
mask_otsu = ~imbinarize(resized_image, threshold);
mask_otsu = imresize(mask_otsu, size(original_image), 'nearest');

figure(1)
imshow(original_image, [])
title('Origin')

figure(2)
imshow(mask_SAM, [])
title('ClearAIM')

figure(3)
imshow(mask_otsu, [])
title('Otsu')

figure(4)
imshow(mask_true, [])
title('Ground True')


%displayImagesWithMasks(paths_pairs)

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

    % List of possible file extensions
    image_extensions = {'*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.gif'};
    mask_extensions = {'*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.gif'};

    % Initialize a structure to store results
    file_pairs = struct('image_path', {}, 'mask_path', {});

    % Initialize a list to store image files
    image_files = [];
    for ext = image_extensions
        image_files = [image_files; dir(fullfile(folder_images, ext{1}))];
    end

    % Initialize a list to store mask files
    mask_files = [];
    for ext = mask_extensions
        mask_files = [mask_files; dir(fullfile(folder_masks, ext{1}))];
    end

    % Map mask names to their full paths for quick lookup
    mask_map = containers.Map();
    for i = 1:length(mask_files)
        [~, mask_name, mask_ext] = fileparts(mask_files(i).name);
        mask_map([mask_name, mask_ext]) = fullfile(folder_masks, mask_files(i).name);
    end

    % Iterate over all image files
    for i = 1:length(image_files)
        % Get the image filename without extension
        [~, filename, ~] = fileparts(image_files(i).name);

        % Construct the corresponding mask name
        mask_name = [filename, mask_suffix];

        % Check all possible mask extensions
        found_mask = false;
        for ext = mask_extensions
            mask_filename = [mask_name, ext{1}(2:end)]; % Remove '*' from extension
            if mask_map.isKey(mask_filename)
                found_mask = true;
                % Add the image and mask paths to the structure
                file_pairs(end+1).image_path = fullfile(folder_images, image_files(i).name);
                file_pairs(end).mask_path = mask_map(mask_filename);
                break;
            end
        end

        % If no mask is found, skip this image
        if ~found_mask
            continue;
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
    area_array = area_array / area_array(1);
    transmittance_array = transmittance_array * 100;
 
    time_x = time_x * 24; % Days to hours

    % Create a figure with a white background
    h = figure;
    set(h, 'Color', 'w');

    % Subplot for Area
    subplot(3, 1, 1);  % 3 rows, 1 column, 1st subplot
    plot(time_x, area_array, 'LineWidth', 2);
    title('Area plot');
    xlabel('Time (hours)');
    ylabel('Normalized Area');
    
    % Subplot for Weber Contrast
    subplot(3, 1, 2);  % 3 rows, 1 column, 2nd subplot
    plot(time_x, web_contrast_array, 'LineWidth', 2);
    title('Weber contrast plot');
    xlabel('Time (hours)');
    ylabel('Weber Contrast');
    
    % Subplot for Transmittance
    subplot(3, 1, 3);  % 3 rows, 1 column, 3rd subplot
    plot(time_x, transmittance_array, 'LineWidth', 2);
    title('Transmittance plot');
    xlabel('Time (hours)');
    ylabel('Transmittance (%)');
end


function time = getImageTime(filename)
    % Function returns the timestamp of the image if available in the EXIF metadata.
    % If the information is not available, it returns an empty array.
    
    % Retrieve the image metadata
    info = imfinfo(filename);
    
    % Initialize the variable 'time' as empty
    time = [];
    
    % Check if the timestamp information exists in the EXIF metadata
    if isfield(info, 'DigitalCamera') && isfield(info.DigitalCamera, 'DateTimeOriginal')
        % Retrieve the timestamp of when the image was taken
        time = info.DigitalCamera.DateTimeOriginal;
    elseif isfield(info, 'FileModDate')
        % Alternatively, use the file modification timestamp
        time = info.FileModDate;
    else
        fprintf('Timestamp metadata is not available for the file: %s\n', filename);
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
    try
       dateTimeObj = datetime(timeStr, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
    catch
       dateTimeObj = datetime(timeStr, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss', 'Locale','pl-PL');
    end
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
    % Create a video writer
    videoWriter = VideoWriter(filename_save, 'MPEG-4');
    videoWriter.Quality = 100;
    open(videoWriter);

    % Preprocess data
    area_array = [metrics_array(:).area] / max([metrics_array(:).area]) * 100; % Normalize area
    web_contrast_array = [metrics_array(:).web_contrast];
    transmittance_array = [metrics_array(:).transmittance] * 100; % Convert to percentage
    times_hours = times_days_array * 24; % Convert days to hours

    % Define y-limits outside the loop for performance
    area_limits = [min(area_array), max(area_array)];
    contrast_limits = [min(web_contrast_array), max(web_contrast_array)];
    transmittance_limits = [min(transmittance_array), max(transmittance_array)];

    % Create the figure and layout
    h_fig = figure('Units', 'Normalized', 'Position', [0 0 1 1]); 
    set(h_fig, 'Color', 'w');

    % Configure tile layout
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for ind = 1:length(paths_pairs)
        path_img = paths_pairs(ind).image_path;
        path_mask = paths_pairs(ind).mask_path;

        % Convert time to hh:mm:ss format
        current_time = seconds(times_hours(ind) * 3600);
        time_string = string(duration(current_time, 'Format', 'hh:mm:ss'));

        % Create a layout with two columns
        subplot('Position', [0.05 0.1 0.5 0.8]); % Image on the left (large)
        displayImageWithMaskContour(path_img, path_mask);
        title(['Image with Mask Contours', ...
            "Elapsed Time [hh:mm:ss]: " + time_string]);
        
        % Plots on the right
        % First plot (area)
        subplot('Position', [0.6 0.54 0.35 0.3]);
        plot(times_hours(1:ind), area_array(1:ind), 'LineWidth', 2, 'Color', 'b');
        title('Normalized Area Over Time [%]');
        xlabel('Time [hours]');
        ylabel('Normalized Area');
        xlim([times_hours(1), times_hours(end)]);
        ylim(area_limits);
        grid on;
        
        % Second plot (transmittance)
        subplot('Position', [0.6 0.15 0.35 0.3]);
        plot(times_hours(1:ind), transmittance_array(1:ind), 'LineWidth', 2, 'Color', '#2e8b57');
        title('Transparency Over Time');
        xlabel('Time [hours]');
        ylabel('Transparency [%]');
        xlim([times_hours(1), times_hours(end)]);
        ylim(transmittance_limits);
        grid on;
    
        % Save the frame to the video
        frameData = getframe(gcf);
        writeVideo(videoWriter, frameData);
    
        % Short pause for visualization (optional)
        pause(0.01);
    end

    % Close the video writer and figure
    close(videoWriter);
    close(h_fig);

    disp(['Animation completed and saved as ', filename_save]);
end

function datetime_obj = filename2datetime(filename)
    % Usunięcie rozszerzenia
    [~, name, ~] = fileparts(filename);
    
    % Konwersja na format daty
    datetime_obj = datetime(name, 'InputFormat', 'yyyy-MM-dd HH-mm-ss');
end


function days_elapsed = getTimeFromName(paths_image)
    time_array = datetime.empty(length(paths_image), 0);
    for ind = 1 : length(paths_image)
        time_array(ind) = filename2datetime(paths_image(ind));
    end
    days_elapsed = days(time_array - min(time_array));
end

function inds = getIndsForTimeSynchronization(time_actual)
    load("wzorcowy_czas.mat");
    inds = zeros(length(times_days_array_original), 1);
    for ind_time = 1 : length(times_days_array_original)
        [~, inds(ind_time)] = min(abs(time_actual - times_days_array_original(ind_time))); 
    end
end
