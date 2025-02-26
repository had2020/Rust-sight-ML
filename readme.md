This project is on hold until pigs fly.
Mainly because Rust is just too new for ML, for me.
I will return, once I learn how to do this in python. 

Pretty much it, it was going to be a ML algorithm, which predicts if an image is pizzia or not.
It failed because I could'nt get a cross entorpy method working on my metal npu, and with other devices I got a whole array of issues, with every single loss method out their for this libary.
Since I'am not a math wiz, combined with a shader king, I will put this project on hold, till I play with to torch or MLX in python :C 

Futher Updates this project will not be finished as the main method to do it is purely cpu based and frameworkless. This project is very important to see the basic formula for a ML network, refer to https://github.com/had2020/RCMLRS for a more up to date rust ML framework that solves my problems. I learned alot from this project on the basics of a ML network and the activation functions.

#to run use
'''rust
cargo run --release
'''

dataset used https://huggingface.co/datasets/nateraw/pizza_not_pizza
