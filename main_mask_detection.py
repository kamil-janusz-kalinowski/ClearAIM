from src.mask_detector import MaskDetectorBuilder

def transform_source_path_to_save_path(path_source: str) -> str:
    return path_source.replace("Materials", "Results")

if __name__ == "__main__":
    builder = MaskDetectorBuilder()
    builder.folderpath_source = r".\Materials\250um brain\skrawek 2"
    builder.folderpath_save = transform_source_path_to_save_path(builder.folderpath_source)
    builder.is_display = True
    
    detector = builder.build()
    detector.process_images()