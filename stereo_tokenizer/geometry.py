"""Shared Student/Depth Anything 3 geometry for rectified RGB inputs."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


GEOMETRY_PREPROCESS_VERSION = "rectified-student-da3-v1"
STUDENT_RESIZE_METHOD = "torch_bilinear_antialias"
DA3_NORMALIZATION = "imagenet_mean_std_v1"
DA3_PATCH_MULTIPLE = 14
_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)


def _positive_hw(value, name):
    try:
        height, width = (int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain exactly two integers") from error
    if height < 1 or width < 1:
        raise ValueError(f"{name} must be positive")
    return height, width


def _nearest_multiple(value: int, multiple: int) -> int:
    down = (value // multiple) * multiple
    up = down + multiple
    return max(1, up if abs(up - value) <= abs(value - down) else down)


@dataclass(frozen=True)
class GeometryMapping:
    """One rectified source mapped independently to Student and DA3 grids."""

    source_hw: tuple[int, int]
    rectified_hw: tuple[int, int]
    student_resized_hw: tuple[int, int]
    student_padding_ltrb: tuple[int, int, int, int]
    student_output_hw: tuple[int, int]
    da3_process_res: int
    da3_process_res_method: str
    da3_processed_hw: tuple[int, int]
    da3_patch_multiple: int = DA3_PATCH_MULTIPLE

    @classmethod
    def create(
        cls,
        rectified_hw,
        *,
        source_hw=None,
        student_output_hw=(256, 256),
        da3_process_res=504,
        da3_process_res_method="upper_bound_resize",
        da3_patch_multiple=DA3_PATCH_MULTIPLE,
    ):
        rectified_h, rectified_w = _positive_hw(rectified_hw, "rectified_hw")
        source_h, source_w = _positive_hw(
            source_hw if source_hw is not None else rectified_hw, "source_hw"
        )
        output_h, output_w = _positive_hw(
            student_output_hw, "student_output_hw"
        )
        process_res = int(da3_process_res)
        patch_multiple = int(da3_patch_multiple)
        if process_res < 1 or patch_multiple < 1:
            raise ValueError("DA3 process resolution and patch multiple must be positive")
        if da3_process_res_method != "upper_bound_resize":
            raise ValueError("geometry contract requires DA3 upper_bound_resize")

        student_scale = min(
            output_h / float(rectified_h), output_w / float(rectified_w)
        )
        student_h = max(1, int(round(rectified_h * student_scale)))
        student_w = max(1, int(round(rectified_w * student_scale)))
        left = (output_w - student_w) // 2
        right = output_w - student_w - left
        top = (output_h - student_h) // 2
        bottom = output_h - student_h - top

        da3_scale = process_res / float(max(rectified_h, rectified_w))
        boundary_h = max(1, int(round(rectified_h * da3_scale)))
        boundary_w = max(1, int(round(rectified_w * da3_scale)))
        da3_h = _nearest_multiple(boundary_h, patch_multiple)
        da3_w = _nearest_multiple(boundary_w, patch_multiple)
        return cls(
            source_hw=(source_h, source_w),
            rectified_hw=(rectified_h, rectified_w),
            student_resized_hw=(student_h, student_w),
            student_padding_ltrb=(left, top, right, bottom),
            student_output_hw=(output_h, output_w),
            da3_process_res=process_res,
            da3_process_res_method=str(da3_process_res_method),
            da3_processed_hw=(da3_h, da3_w),
            da3_patch_multiple=patch_multiple,
        )

    def to_metadata(self):
        return {
            "preprocess_version": GEOMETRY_PREPROCESS_VERSION,
            "source_hw": list(self.source_hw),
            "rectified_hw": list(self.rectified_hw),
            "student_resize_method": STUDENT_RESIZE_METHOD,
            "student_resized_hw": list(self.student_resized_hw),
            "student_padding_ltrb": list(self.student_padding_ltrb),
            "student_output_hw": list(self.student_output_hw),
            "da3_process_res": self.da3_process_res,
            "da3_process_res_method": self.da3_process_res_method,
            "da3_processed_hw": list(self.da3_processed_hw),
            "da3_patch_multiple": self.da3_patch_multiple,
            "da3_normalization": DA3_NORMALIZATION,
        }

    def to_collatable_metadata(self):
        metadata = self.to_metadata()
        for key in (
            "source_hw",
            "rectified_hw",
            "student_resized_hw",
            "student_padding_ltrb",
            "student_output_hw",
            "da3_processed_hw",
        ):
            metadata[key] = torch.tensor(metadata[key], dtype=torch.int64)
        return metadata

    @classmethod
    def from_metadata(cls, metadata):
        if not isinstance(metadata, dict):
            raise ValueError("geometry_mapping must be a dictionary")
        required = {
            "preprocess_version",
            "source_hw",
            "rectified_hw",
            "student_resize_method",
            "student_resized_hw",
            "student_padding_ltrb",
            "student_output_hw",
            "da3_process_res",
            "da3_process_res_method",
            "da3_processed_hw",
            "da3_patch_multiple",
            "da3_normalization",
        }
        if set(metadata) != required:
            raise ValueError("geometry_mapping fields mismatch")
        mapping = cls.create(
            metadata["rectified_hw"],
            source_hw=metadata["source_hw"],
            student_output_hw=metadata["student_output_hw"],
            da3_process_res=metadata["da3_process_res"],
            da3_process_res_method=metadata["da3_process_res_method"],
            da3_patch_multiple=metadata["da3_patch_multiple"],
        )
        if metadata != mapping.to_metadata():
            raise ValueError("geometry_mapping values do not match derived geometry")
        return mapping

    @classmethod
    def from_collated(cls, metadata, batch_size):
        if not isinstance(metadata, dict):
            raise ValueError("collated geometry_mapping must be a dictionary")

        def uniform_string(name):
            values = metadata.get(name)
            if isinstance(values, str):
                values = [values]
            values = list(values or ())
            if len(values) != batch_size or any(value != values[0] for value in values):
                raise ValueError(f"geometry batch must contain one {name}")
            return values[0]

        def uniform_tensor(name, width=None):
            value = metadata.get(name)
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            expected = (batch_size,) if width is None else (batch_size, width)
            if tuple(value.shape) != expected:
                raise ValueError(f"collated geometry {name} shape mismatch")
            if batch_size > 1 and not torch.equal(value, value[0:1].expand_as(value)):
                raise ValueError(f"geometry batch must contain one {name}")
            first = value[0].detach().cpu()
            return int(first.item()) if width is None else first.tolist()

        normalized = {
            "preprocess_version": uniform_string("preprocess_version"),
            "source_hw": uniform_tensor("source_hw", 2),
            "rectified_hw": uniform_tensor("rectified_hw", 2),
            "student_resize_method": uniform_string("student_resize_method"),
            "student_resized_hw": uniform_tensor("student_resized_hw", 2),
            "student_padding_ltrb": uniform_tensor("student_padding_ltrb", 4),
            "student_output_hw": uniform_tensor("student_output_hw", 2),
            "da3_process_res": uniform_tensor("da3_process_res"),
            "da3_process_res_method": uniform_string("da3_process_res_method"),
            "da3_processed_hw": uniform_tensor("da3_processed_hw", 2),
            "da3_patch_multiple": uniform_tensor("da3_patch_multiple"),
            "da3_normalization": uniform_string("da3_normalization"),
        }
        return cls.from_metadata(normalized)

    def student_letterbox(self, rectified_rgb):
        if rectified_rgb.ndim != 4 or tuple(rectified_rgb.shape[1:]) != (
            3,
            *self.rectified_hw,
        ):
            raise ValueError("rectified RGB tensor disagrees with geometry metadata")
        resized = F.interpolate(
            rectified_rgb.float(),
            size=self.student_resized_hw,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        left, top, right, bottom = self.student_padding_ltrb
        output = F.pad(resized, (left, right, top, bottom), value=0.0)
        mask = torch.zeros(
            (rectified_rgb.shape[0], 1, *self.student_output_hw), dtype=torch.bool
        )
        mask[
            :,
            :,
            top : self.student_output_hw[0] - bottom,
            left : self.student_output_hw[1] - right,
        ] = True
        return output, mask

    def da3_preprocess(self, rectified_rgb):
        if rectified_rgb.ndim != 4 or tuple(rectified_rgb.shape[1:]) != (
            3,
            *self.rectified_hw,
        ):
            raise ValueError("rectified RGB tensor disagrees with geometry metadata")
        if rectified_rgb.dtype != torch.uint8:
            raise TypeError("DA3 worker preprocessing requires uint8 RGB")
        source = rectified_rgb.detach().cpu().numpy()
        output = []
        target_h, target_w = self.da3_processed_hw
        rectified_h, rectified_w = self.rectified_hw
        scale = self.da3_process_res / float(max(rectified_h, rectified_w))
        boundary_h = max(1, int(round(rectified_h * scale)))
        boundary_w = max(1, int(round(rectified_w * scale)))
        first_interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        for frame in source:
            image = np.ascontiguousarray(frame.transpose(1, 2, 0))
            if (boundary_h, boundary_w) != self.rectified_hw:
                image = cv2.resize(
                    image,
                    (boundary_w, boundary_h),
                    interpolation=first_interpolation,
                )
            if (target_h, target_w) != (boundary_h, boundary_w):
                upscale = target_h > boundary_h or target_w > boundary_w
                image = cv2.resize(
                    image,
                    (target_w, target_h),
                    interpolation=cv2.INTER_CUBIC if upscale else cv2.INTER_AREA,
                )
            output.append(torch.from_numpy(image).permute(2, 0, 1))
        tensor = torch.stack(output).float().div_(255.0)
        mean = _IMAGENET_MEAN[:, None, None]
        std = _IMAGENET_STD[:, None, None]
        return tensor.sub_(mean).div_(std)

    def map_da3_output_to_student(self, value):
        if value.ndim != 4 or tuple(value.shape[-2:]) != self.da3_processed_hw:
            raise ValueError("DA3 output tensor disagrees with geometry metadata")
        batch, frames = value.shape[:2]
        resized = F.interpolate(
            value.reshape(batch * frames, 1, *self.da3_processed_hw),
            size=self.student_resized_hw,
            mode="bilinear",
            align_corners=False,
        )
        left, top, right, bottom = self.student_padding_ltrb
        padded = F.pad(resized, (left, right, top, bottom), value=0.0)
        return padded.reshape(
            batch, frames, 1, *self.student_output_hw
        ).permute(0, 2, 1, 3, 4)
