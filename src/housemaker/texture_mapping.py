# ### Imports ###
from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter

from housemaker.models import WallTextureData

# ### Texture painting helpers ###
def paint_wall_texture_crop(
    painter: QPainter,
    texture_data: WallTextureData,
    target_rect: QRectF,
) -> bool:
    source_image = QImage(texture_data.image_path)
    if source_image.isNull():
        return False

    source_rect = _clamp_source_rect(texture_data, source_image)
    if source_rect.width() <= 0.0 or source_rect.height() <= 0.0:
        return False

    painter.drawImage(target_rect, source_image, source_rect)
    return True


# ### Geometry helpers ###
def _clamp_source_rect(
    texture_data: WallTextureData,
    source_image: QImage,
) -> QRectF:
    image_width = max(1.0, float(source_image.width()))
    image_height = max(1.0, float(source_image.height()))
    source_width = min(max(1.0, float(texture_data.source_width)), image_width)
    source_height = min(max(1.0, float(texture_data.source_height)), image_height)
    source_x = min(
        max(0.0, float(texture_data.source_x)),
        max(0.0, image_width - source_width),
    )
    source_y = min(
        max(0.0, float(texture_data.source_y)),
        max(0.0, image_height - source_height),
    )
    return QRectF(source_x, source_y, source_width, source_height)
