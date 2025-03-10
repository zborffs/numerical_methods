% 1a.
syms p q real
syms pn qn real
syms m omega real
syms h real
H = p^2 / (2*m) + m * omega^2 * q^2 / 2
[pn1, qn1] = stoermer_verlet(H, p, q, pn, qn, h);
a1 = coeffs(pn1, [qn, pn])
a2 = coeffs(qn1, [qn, pn])
A = [a1; a2]


% 1b.
J = [0 1; -1 0];
simplify(A' * J * A - J) % verifies A' J A = J


% 1c.
Hshadow = p^2 / (2*m) + 1/2 * m * omega^2 * q^2 * (1 - (omega * h / 2)^2)
Hn = subs(subs(Hshadow, p, pn), q, qn)
Hn1 = subs(subs(Hshadow, p, pn1), q, qn1)
simplify(Hn1 - Hn) % checked! 



%% 2a.
syms u v x y real
H = 1/2 * u^2 + 1/2 * v^2 - 1 / sqrt(x^2 + y^2);
p = [u, v];
q = [x, y];
pn = sym('pn', [1 2]);
qn = sym('qn', [1 2]);
[pn_1, qn_1] = stoermer_verlet(H, p, q, pn, qn, h);
% pn_11 = pn_1(1)
% pn_12 = pn_1(2)
% qn_11 = qn_1(1)
% qn_12 = qn_1(2)
% a1 = coeffs(pn_11, [qn, pn])
% a2 = coeffs(pn_12, [qn, pn])
% a3 = coeffs(qn_11, [qn, pn])
% a4 = coeffs(qn_12, [qn, pn])
% A = simplify([a1; a2; a3; a4])


function [pn_1, qn_1] = stoermer_verlet(H, p, q, pn, qn, h)

    pn_12 = simplify(pn - h/2 * subs(jacobian(H, q), q, qn))
    qn_1 = simplify(qn + h * (subs(jacobian(H, p), p, pn_12)))
    pn_1 = simplify(pn_12 - h /2 * subs(jacobian(H, q), q, qn_1))

end