import cv2 as cv

def draw_label_with_bg(
    img,
    text,
    topleft,
    font_scale: float = 1.0,
    text_color=(255, 255, 255),
    bg_color=(255, 100, 100),
):
    font = cv.FONT_HERSHEY_SIMPLEX
    ((w, h), _) = cv.getTextSize(text, font, fontScale=font_scale, thickness=1)
    x, y = topleft
    cv.rectangle(img, (x, y - h - 4), (x + w, y + 2), bg_color, -1)
    cv.putText(
        img,
        text,
        (x, y - 2),
        font,
        font_scale,
        text_color,
        thickness=1,
        lineType=cv.LINE_AA,
    )
