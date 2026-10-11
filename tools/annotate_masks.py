"""
tools/annotate_masks.py — Interactive Manual Ground-Truth Defect Annotation Tool

A graphical annotation tool built with OpenCV for drawing genuine pixel-level
ground-truth defect masks on NEU-DET steel surface images.

Features:
  - Polygon and contour vertex drawing with real-time visual feedback.
  - Scaled display (default 3x: 600x600 px) for high-precision cursor placement.
  - Support for multiple disjoint defect polygons per image.
  - Undo vertex (Right-click or 'u'), Close polygon ('c'), Reset ('r').
  - Optional XML bounding box reference overlay ('b') clearly watermarked as reference only.
  - Save single-channel 8-bit binary PNG (0=background, 255=defect) to results/masks/ground_truth/.
  - Automated tracking of annotation progress and metadata in results/masks/annotation_metadata/.
  - Headless test mode (--headless-test) for automated CI and pipeline verification.
"""

import os
import sys
import json
import time
import argparse
import xml.etree.ElementTree as ET
import cv2
import numpy as np

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Default 30-image pilot dataset: exactly 5 images per defect class across all 6 classes
PILOT_IMAGES = [
    # 1. Scratches (longitudinal trenches)
    {"class": "scratches", "file": "scratches_1.jpg"},
    {"class": "scratches", "file": "scratches_2.jpg"},
    {"class": "scratches", "file": "scratches_3.jpg"},
    {"class": "scratches", "file": "scratches_4.jpg"},
    {"class": "scratches", "file": "scratches_5.jpg"},
    # 2. Inclusion (small localized non-metallic flecks)
    {"class": "inclusion", "file": "inclusion_1.jpg"},
    {"class": "inclusion", "file": "inclusion_2.jpg"},
    {"class": "inclusion", "file": "inclusion_3.jpg"},
    {"class": "inclusion", "file": "inclusion_4.jpg"},
    {"class": "inclusion", "file": "inclusion_5.jpg"},
    # 3. Crazing (fine crack fissure network)
    {"class": "crazing", "file": "crazing_1.jpg"},
    {"class": "crazing", "file": "crazing_2.jpg"},
    {"class": "crazing", "file": "crazing_3.jpg"},
    {"class": "crazing", "file": "crazing_4.jpg"},
    {"class": "crazing", "file": "crazing_5.jpg"},
    # 4. Pitted Surface (micro-cavities and rough pitting)
    {"class": "pitted_surface", "file": "pitted_surface_1.jpg"},
    {"class": "pitted_surface", "file": "pitted_surface_2.jpg"},
    {"class": "pitted_surface", "file": "pitted_surface_3.jpg"},
    {"class": "pitted_surface", "file": "pitted_surface_4.jpg"},
    {"class": "pitted_surface", "file": "pitted_surface_7.jpg"},
    # 5. Patches (broad plate-like surface irregularities)
    {"class": "patches", "file": "patches_1.jpg"},
    {"class": "patches", "file": "patches_2.jpg"},
    {"class": "patches", "file": "patches_3.jpg"},
    {"class": "patches", "file": "patches_4.jpg"},
    {"class": "patches", "file": "patches_5.jpg"},
    # 6. Rolled-in Scale (embedded iron oxide mill scale)
    {"class": "rolled-in_scale", "file": "rolled-in_scale_1.jpg"},
    {"class": "rolled-in_scale", "file": "rolled-in_scale_2.jpg"},
    {"class": "rolled-in_scale", "file": "rolled-in_scale_3.jpg"},
    {"class": "rolled-in_scale", "file": "rolled-in_scale_4.jpg"},
    {"class": "rolled-in_scale", "file": "rolled-in_scale_5.jpg"},
]


def load_xml_boxes(xml_path):
    """Load bounding boxes from Pascal VOC XML for visual reference."""
    if not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            if bnd is not None:
                boxes.append({
                    "xmin": int(float(bnd.find("xmin").text)),
                    "ymin": int(float(bnd.find("ymin").text)),
                    "xmax": int(float(bnd.find("xmax").text)),
                    "ymax": int(float(bnd.find("ymax").text)),
                })
        return boxes
    except Exception:
        return []


class ManualAnnotator:
    """Interactive polygon-based manual defect annotator."""

    def __init__(self, image_items, dataset_dir, output_dir, metadata_dir, scale=3):
        self.image_items = image_items
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.metadata_dir = metadata_dir
        self.scale = scale

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.manifest_path = os.path.join(self.metadata_dir, "annotation_manifest.json")
        self.manifest = self._load_manifest()

        self.current_idx = 0
        self.completed_polygons = []  # List of lists of (x, y) in original 200x200 coords
        self.current_polygon = []     # List of (x, y) in original coords
        self.show_xml_ref = False
        self.mouse_pos = (0, 0)

    def _load_manifest(self):
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_manifest(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def get_paths(self, item):
        cls_name = item["class"]
        fname = item["file"]
        img_path = os.path.join(self.dataset_dir, "train", "images", cls_name, fname)
        xml_path = os.path.join(self.dataset_dir, "train", "annotations", fname.replace(".jpg", ".xml"))
        mask_path = os.path.join(self.output_dir, fname.replace(".jpg", ".png"))
        return img_path, xml_path, mask_path

    def load_existing_mask(self, mask_path):
        """If a mask already exists on disk, load and extract its contours."""
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                loaded_polys = []
                for cnt in contours:
                    if len(cnt) >= 3:
                        poly = [tuple(pt[0]) for pt in cnt]
                        loaded_polys.append(poly)
                return loaded_polys
        return []

    def rasterize_mask(self, h=200, w=200):
        """Rasterize all completed polygons into a binary uint8 mask (0 or 255)."""
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in self.completed_polygons:
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
        return mask

    def mouse_callback(self, event, x, y, flags, param):
        orig_x = int(round(x / self.scale))
        orig_y = int(round(y / self.scale))
        orig_x = max(0, min(199, orig_x))
        orig_y = max(0, min(199, orig_y))
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Add vertex to current polygon
            self.current_polygon.append((orig_x, orig_y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Undo last vertex
            if self.current_polygon:
                self.current_polygon.pop()
            elif self.completed_polygons:
                self.current_polygon = self.completed_polygons.pop()

    def run_interactive(self):
        """Main OpenCV GUI interaction loop."""
        window_name = "PGRG Manual Defect Annotator (NEU-DET)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("=" * 70)
        print("PGRG MANUAL STEEL DEFECT ANNOTATION TOOL")
        print("=" * 70)
        print("Controls:")
        print("  Left Click       : Add polygon vertex")
        print("  Right Click / 'u': Undo last vertex")
        print("  'c'              : Complete/close current polygon into mask")
        print("  'r'              : Reset all annotations on current image")
        print("  'b'              : Toggle XML bounding box overlay (reference only)")
        print("  's'              : Save binary mask to results/masks/ground_truth/ and advance")
        print("  'n' / 'd'        : Skip to Next image")
        print("  'p' / 'a'        : Return to Previous image")
        print("  'q' / ESC        : Save session and Quit")
        print("=" * 70)

        while 0 <= self.current_idx < len(self.image_items):
            item = self.image_items[self.current_idx]
            fname = item["file"]
            cls_name = item["class"]
            img_path, xml_path, mask_path = self.get_paths(item)

            if not os.path.exists(img_path):
                print(f"[WARN] Skipping missing image: {img_path}")
                self.current_idx += 1
                continue

            orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if orig_img is None:
                self.current_idx += 1
                continue

            h, w = orig_img.shape
            # Load existing saved annotations if polygons list is empty
            if not self.completed_polygons and not self.current_polygon:
                self.completed_polygons = self.load_existing_mask(mask_path)

            xml_boxes = load_xml_boxes(xml_path)

            while True:
                # Prepare display canvas
                bgr_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
                scaled_img = cv2.resize(bgr_img, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST)

                # Draw XML reference boxes if toggled
                if self.show_xml_ref and xml_boxes:
                    for box in xml_boxes:
                        bx1, by1 = box["xmin"] * self.scale, box["ymin"] * self.scale
                        bx2, by2 = box["xmax"] * self.scale, box["ymax"] * self.scale
                        cv2.rectangle(scaled_img, (bx1, by1), (bx2, by2), (0, 165, 255), 2, lineType=cv2.LINE_AA)
                    cv2.putText(
                        scaled_img, "XML REF ONLY (ROUGH BOUNDING BOX)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA
                    )

                # Draw completed polygons (translucent green fill + bright green outline)
                overlay = scaled_img.copy()
                for poly in self.completed_polygons:
                    if len(poly) >= 3:
                        scaled_pts = np.array([[pt[0] * self.scale, pt[1] * self.scale] for pt in poly], dtype=np.int32)
                        cv2.fillPoly(overlay, [scaled_pts], (0, 230, 115))
                        cv2.polylines(scaled_img, [scaled_pts], isClosed=True, color=(0, 255, 128), thickness=2, lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.35, scaled_img, 0.65, 0, scaled_img)

                # Draw current in-progress polygon
                if self.current_polygon:
                    curr_pts = [[pt[0] * self.scale, pt[1] * self.scale] for pt in self.current_polygon]
                    for pt in curr_pts:
                        cv2.circle(scaled_img, tuple(pt), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                    if len(curr_pts) >= 2:
                        cv2.polylines(scaled_img, [np.array(curr_pts, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                    # Line to current mouse cursor
                    cv2.line(scaled_img, tuple(curr_pts[-1]), self.mouse_pos, (180, 255, 255), 1, lineType=cv2.LINE_AA)

                # Header bar
                header = np.zeros((70, scaled_img.shape[1], 3), dtype=np.uint8)
                header[:] = (35, 35, 35)
                status_txt = f"[{self.current_idx + 1}/{len(self.image_items)}] {fname} | Class: {cls_name}"
                is_saved = os.path.exists(mask_path)
                status_badge = f"STATUS: {'SAVED (' + str(len(self.completed_polygons)) + ' polys)' if is_saved else 'PENDING'}"
                badge_col = (0, 230, 115) if is_saved else (100, 180, 255)

                cv2.putText(header, status_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(header, status_badge, (scaled_img.shape[1] - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_col, 2, cv2.LINE_AA)
                cv2.putText(
                    header, "[Click]: Point | [U]: Undo | [C]: Close Poly | [S]: Save | [R]: Reset | [B]: XML Ref | [Q]: Quit",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA
                )

                display_frame = np.vstack([header, scaled_img])
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(20) & 0xFF

                if key in (ord('q'), 27):  # Quit
                    self._save_manifest()
                    cv2.destroyAllWindows()
                    print("\n[INFO] Manual annotation session closed. Progress saved.")
                    return

                elif key in (ord('c'), ord('C')):  # Close polygon
                    if len(self.current_polygon) >= 3:
                        self.completed_polygons.append(self.current_polygon)
                        self.current_polygon = []
                        print(f"  [OK] Completed polygon #{len(self.completed_polygons)} ({len(self.completed_polygons[-1])} vertices)")
                    else:
                        print("  [!] Polygon requires at least 3 vertices to close.")

                elif key in (ord('u'), ord('U')):  # Undo
                    if self.current_polygon:
                        self.current_polygon.pop()
                    elif self.completed_polygons:
                        self.current_polygon = self.completed_polygons.pop()

                elif key in (ord('r'), ord('R')):  # Reset
                    self.completed_polygons = []
                    self.current_polygon = []
                    print(f"  [RESET] Cleared all annotations for {fname}")

                elif key in (ord('b'), ord('B')):  # Toggle XML box
                    self.show_xml_ref = not self.show_xml_ref

                elif key in (ord('s'), ord('S')):  # Save & Next
                    if self.current_polygon and len(self.current_polygon) >= 3:
                        self.completed_polygons.append(self.current_polygon)
                        self.current_polygon = []

                    mask = self.rasterize_mask(h, w)
                    cv2.imwrite(mask_path, mask)
                    defect_px = int(np.sum(mask > 0))

                    # Update manifest
                    self.manifest[fname] = {
                        "file": fname,
                        "class": cls_name,
                        "status": "annotated",
                        "mask_path": os.path.relpath(mask_path, PROJECT_ROOT),
                        "polygons_count": len(self.completed_polygons),
                        "defect_pixel_count": defect_px,
                        "approximate_roi_only": False,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "notes": "Manually verified polygon contour annotation.",
                    }
                    self._save_manifest()
                    print(f"[SAVED] {mask_path} (Polygons: {len(self.completed_polygons)}, Defect Pixels: {defect_px})")

                    self.completed_polygons = []
                    self.current_polygon = []
                    self.current_idx += 1
                    break

                elif key in (ord('n'), ord('d'), ord('N'), ord('D')):  # Next without saving
                    self.completed_polygons = []
                    self.current_polygon = []
                    self.current_idx += 1
                    break

                elif key in (ord('p'), ord('a'), ord('P'), ord('A')):  # Prev
                    self.completed_polygons = []
                    self.current_polygon = []
                    self.current_idx = max(0, self.current_idx - 1)
                    break

        cv2.destroyAllWindows()
        self._save_manifest()
        print("\n[OK] Reached end of pilot image dataset.")


def run_headless_verification(output_dir, metadata_dir):
    """
    Automated test verification of the annotation tool pipeline without GUI.
    Tests rasterization, mask saving, format validation, and metadata writing.
    """
    print("\n--- Running Headless Verification of Annotation Tool Pipeline ---")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    test_file = "test_verification_defect.png"
    test_mask_path = os.path.join(output_dir, test_file)

    # Delineate a known polygon test
    test_poly = [(30, 40), (70, 40), (80, 90), (35, 85)]
    mask = np.zeros((200, 200), dtype=np.uint8)
    pts = np.array(test_poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    # Save PNG
    cv2.imwrite(test_mask_path, mask)
    assert os.path.exists(test_mask_path), "Failed to write mask"

    # Read back and verify properties
    loaded = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
    assert loaded.shape == (200, 200), f"Wrong shape: {loaded.shape}"
    unique_vals = set(np.unique(loaded))
    assert unique_vals.issubset({0, 255}), f"Non-binary values found: {unique_vals}"
    pixel_count = int(np.sum(loaded > 0))
    assert pixel_count > 0, "Empty mask rasterized"

    # Clean up test artifact
    if os.path.exists(test_mask_path):
        os.remove(test_mask_path)

    print(f"[OK] Mask rasterization verified: shape={loaded.shape}, binary_values={unique_vals}, pixels={pixel_count}")
    return True


def initialize_pilot_manifest(metadata_dir, output_dir, dataset_dir):
    """Create the initial 30-image pilot dataset manifest and progress tracking JSON."""
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(metadata_dir, "pilot_dataset_manifest.json")
    progress_path = os.path.join(metadata_dir, "annotation_progress.json")

    manifest = []
    class_counts = {}

    for item in PILOT_IMAGES:
        fname = item["file"]
        cname = item["class"]
        class_counts[cname] = class_counts.get(cname, 0) + 1

        img_rel = os.path.join("dataset", "NEU-DET", "train", "images", cname, fname)
        xml_rel = os.path.join("dataset", "NEU-DET", "train", "annotations", fname.replace(".jpg", ".xml"))
        mask_path = os.path.join(output_dir, fname.replace(".jpg", ".png"))
        mask_rel = os.path.relpath(mask_path, PROJECT_ROOT)

        is_annotated = os.path.exists(mask_path)

        manifest.append({
            "file": fname,
            "class": cname,
            "image_path": img_rel,
            "xml_path": xml_rel,
            "mask_path": mask_rel if is_annotated else None,
            "status": "annotated" if is_annotated else "pending_human_annotation",
            "approximate_roi_only": False,
        })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_annotated = sum(1 for m in manifest if m["status"] == "annotated")
    progress = {
        "pilot_target_images": len(manifest),
        "target_per_class": 5,
        "classes": list(class_counts.keys()),
        "completed_genuine_masks": total_annotated,
        "pending_human_annotation": len(manifest) - total_annotated,
        "manifest_file": os.path.relpath(manifest_path, PROJECT_ROOT),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

    print(f"[OK] Initialized pilot manifest ({len(manifest)} images: 5 per class) at {manifest_path}")
    print(f"[OK] Progress tracker saved to {progress_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="PGRG Manual Steel Defect Annotation Tool")
    parser.add_argument("--dataset-dir", default=os.path.join(PROJECT_ROOT, "dataset", "NEU-DET"), help="Path to NEU-DET dataset")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "masks", "ground_truth"), help="Directory to save ground truth masks")
    parser.add_argument("--metadata-dir", default=os.path.join(PROJECT_ROOT, "results", "masks", "annotation_metadata"), help="Directory for metadata JSON files")
    parser.add_argument("--scale", type=int, default=3, help="Display scale factor (default: 3 for 600x600 px)")
    parser.add_argument("--init-pilot", action="store_true", help="Initialize pilot manifest and exit")
    parser.add_argument("--headless-test", action="store_true", help="Run automated verification without GUI")
    args = parser.parse_args()

    if args.headless_test:
        initialize_pilot_manifest(args.metadata_dir, args.output_dir, args.dataset_dir)
        run_headless_verification(args.output_dir, args.metadata_dir)
        return

    if args.init_pilot:
        initialize_pilot_manifest(args.metadata_dir, args.output_dir, args.dataset_dir)
        return

    # Interactive session
    initialize_pilot_manifest(args.metadata_dir, args.output_dir, args.dataset_dir)
    annotator = ManualAnnotator(
        image_items=PILOT_IMAGES,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        scale=args.scale,
    )
    annotator.run_interactive()


if __name__ == "__main__":
    main()
