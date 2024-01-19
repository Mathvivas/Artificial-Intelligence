from torchvision import transforms
import module as mo


model = mo.load_model('model.pth')
class_labels = ['hello', 'yes', 'no', 'thanks', 'i love you']

mo.predict_on_webcam(model, class_labels, transforms.ToTensor())
