%% Swift-Hohenberg hallucination patterns (Bressloff)
% du/dt = r u - (k0^2 + lap)^2 u - u^3, where lap is the Laplacian.
% This is the reduced pattern-forming equation Bressloff derives for
% cortical hallucination patterns.

clear; close all; clc;

%% Parameters
N   = 160;       % grid size (NxN)
L   = 40;        % physical domain size
r   = 0.4;       % distance past the instability
k0  = 1.0;       % preferred spatial frequency
dt  = 0.1;       % time step
T   = 200;       % total time
nSteps = round(T/dt);

%% Spatial grid (periodic)
x  = linspace(-L/2, L/2, N+1); x(end) = [];  % drop duplicate
y  = linspace(-L/2, L/2, N+1); y(end) = [];
[~, ~] = meshgrid(x, y);  %#ok<ASGLU>   % not used

dx = x(2) - x(1);
dy = y(2) - y(1);

% Fourier wavenumbers
kx = (2*pi/L) * [0:(N/2-1), -N/2:-1];
ky = kx;
[KX, KY] = meshgrid(kx, ky);
k2 = KX.^2 + KY.^2;

% linear operator in Fourier space: L(k) = r - (k0^2 - k^2)^2
Lk = r - (k0^2 - k2).^2;

%% Initial condition: small random noise
u = 0.1 * randn(N, N);

%% Figure and GIF path
figure('Color','w','Position',[200 200 800 700]);
colormap(parula);

homeDir = getenv('HOME');
gifPath = fullfile(homeDir, 'Desktop', 'Bressloff_SH_Pattern.gif');

%% Time stepping, semi-implicit in Fourier space
for step = 1:nSteps

    % -u^3 treated explicitly
    rhs   = u - dt * (u.^3);
    rhs_k = fft2(rhs);

    % solve (1 - dt*Lk) * u_{n+1}(k) = rhs_k
    u_k_new = rhs_k ./ (1 - dt * Lk);
    u       = real(ifft2(u_k_new));

    % plot every 10 steps
    if mod(step, 10) == 0 || step == nSteps
        imagesc(x, y, u);
        axis image off;
        title(sprintf('Swift-Hohenberg cortical pattern  (t = %.1f)', step*dt), ...
              'FontSize', 18, 'FontWeight', 'bold');
        colorbar;

        drawnow;

        % add to GIF
        frame = getframe(gcf);
        [imind, cm] = rgb2ind(frame2im(frame), 256);

        if step == 10
            imwrite(imind, cm, gifPath, ...
                    'gif', 'Loopcount', inf, 'DelayTime', 0.08);
        else
            imwrite(imind, cm, gifPath, ...
                    'gif', 'WriteMode', 'append', 'DelayTime', 0.08);
        end
    end
end

fprintf('\nGIF saved to:\n%s\n', gifPath);
v = VideoWriter('SH_pattern.mp4','MPEG-4');
v.FrameRate = 20;
open(v);

for t = 1:300
    u = evolveSH(u, dt, dx, dy);
    imagesc(u); axis off equal;
    colormap(parula);
    drawnow;

    writeVideo(v, getframe(gcf));
end

close(v);
