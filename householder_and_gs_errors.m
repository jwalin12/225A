clear;
close all;

mag = magic(7);
h = hilb(7);

[R, U] = house_qr(mag);
I = eye(size(U));
mag_house = house_apply(U,I);
mag_gs = gram_schmidt(mag);
[R, U] = house_qr(h);
I = eye(size(U));
hilb_house = house_apply(U,I);
hilb_gs = gram_schmidt(h);

mag_house_factor =   norm(mag - (mag*inv(mag_house.'))*mag_house.', inf)/norm(mag_house, inf);
hilb_house_factor = norm(mag - (mag*inv(hilb_house.'))*hilb_house.', inf)/norm(hilb_house, inf);
mag_gs_factor =   norm(mag - (mag*inv(mag_gs.'))*mag_gs.', inf)/norm(mag_gs, inf);
hilb_gs_factor = norm(mag - (mag*inv(hilb_gs.'))*hilb_gs.', inf)/norm(hilb_gs, inf);
display(mag_house_factor);
display(hilb_house_factor);
display(mag_gs_factor);
display(hilb_gs_factor);
mag_house_orth = norm(eye(7) - mag_house*mag_house.', inf);
hilb_house_orth = norm(eye(7) - hilb_house*hilb_house.', inf);
mag_gs_orth = norm(eye(7) - mag_gs*mag_gs.', inf);
hilb_gs_orth = norm(eye(7) - hilb_gs*hilb_gs.', inf);
display(mag_house_orth);
display(hilb_house_orth);
display(mag_gs_orth);
display(hilb_gs_orth);










function U = gram_schmidt(V)
    n = size(V, 1);
    k = size(V, 2);
    U = zeros(n, k);
    U(:, 1) = V(:, 1) / sqrt(V(:, 1)'*V(:,1));
    for i = 2:k
        U(:, i) = V(:, i);
        for j = 1:i - 1
            U(:, i) = U(:, i) - (U(:, j)'*U(:,i) )/( U(:,j)' * U(:, j)) * U(:, j);
        end
        U(:, i) = U(:, i) / sqrt(U(:, i)'*U(:,i));
    end
end


function u = house_gen(x)
    % u = house_gen(x)
    % Generate Householder reflection.
    % u = house_gen(x) returns u with norm(u) = sqrt(2), and
    % H(u,x) = x - u*(u'*x) = -+ norm(x)*e_1.
    
    % Modify the sign function so that sign(0) = 1.
    sig = @(u) sign(u) + (u==0);
    
    nu = norm(x);
    if nu ~= 0
        u = x/nu;
        u(1) = u(1) + sig(u(1));
        u = u/sqrt(abs(u(1)));
    else
        u = x;
        u(1) = sqrt(2);
    end
end


function [R,U] = house_qr(A)
    % Householder reflections for QR decomposition.
    % [R,U] = house_qr(A) returns
    % R, the upper triangular factor, and
    % U, the reflector generators for use by house_apply.    
    H = @(u,x) x - u*(u'*x);
    [m,n] = size(A);
    U = zeros(m,n);
    R = A;
    for j = 1:min(m,n)
        u = house_gen(R(j:m,j));
        U(j:m,j) = u;
        R(j:m,j:n) = H(u,R(j:m,j:n));
        R(j+1:m,j) = 0;
    end
end



function Z = house_apply(U,X)
    % Apply Householder reflections.
    % Z = house_apply(U,X), with U from house_qr
    % computes Q*X without actually computing Q.
    H = @(u,x) x - u*(u'*x);
    Z = X;
    [~,n] = size(U);
    for j = n:-1:1
        Z = H(U(:,j),Z);
    end
end

function Z = house_apply_transpose(U,X)
    % Apply Householder transposed reflections.
    % Z = house_apply(U,X), with U from house_qr
    % computes Q'*X without actually computing Q'.
    H = @(u,x) x - u*(u'*x);
    Z = X;
    [~,n] = size(U);
    for j = 1:n
        Z = H(U(:,j),Z);
    end
end
