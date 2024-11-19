from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField


class ImageForm(FlaskForm):
    image = FileField('Upload X-ray Image', validators=[
        FileRequired('File is required'),
        FileAllowed(['png', 'jpg', 'jpeg'], 'Only X-ray images (PNG, JPG, JPEG) are allowed')
    ], render_kw={"class": "form-control mb-3"})  # Bootstrap classes for styling

    submit = SubmitField('Submit', render_kw={"class": "btn btn-primary mt-3"})  # Bootstrap button styling
