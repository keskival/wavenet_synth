load("sound.mat")
#load("image.mat")
subplot(1,1,1)
size(s)
seed_length = 1024 * 512;
s = s(:,seed_length-500:size(s)(2));
plot(s)
soundsc(s, 48000)
#subplot(1, 2, 2)
#imagesc(flipud(i'))
wavwrite(s, 48000, "generated.wav")
