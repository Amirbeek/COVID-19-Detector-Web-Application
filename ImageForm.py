from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField


class ImageForm(FlaskForm):
    image = FileField('Upload   Image', validators=[
        FileRequired('File is required'),
        FileAllowed(['png', 'jpg', 'jpeg'], 'Only X-ray images (PNG, JPG, JPEG) are allowed')
    ], render_kw={"class": "form-control d-none hop", "x-ref": "file"})

    submit = SubmitField('Submit', render_kw={"class": "btn btn-success mt-3"})
