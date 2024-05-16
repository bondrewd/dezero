const std = @import("std");
const testing = std.testing;

const DeZeroError = error{
    BackwardBeforeForward,
};

const Variable = struct {
    data: f32,
    grad: ?f32 = null,
};

const Square = struct {
    input: ?Variable = null,

    const Self = @This();

    pub fn forward(self: *Self, x: Variable) Variable {
        self.input = x;
        return .{
            .data = x.data * x.data,
        };
    }

    pub fn backward(self: Self, gy: f32) DeZeroError!f32 {
        const x = self.input orelse return DeZeroError.BackwardBeforeForward;
        const gx = 2 * x.data * gy;
        return gx;
    }
};

const Exp = struct {
    input: ?Variable = null,

    const Self = @This();

    pub fn forward(self: *Self, x: Variable) Variable {
        self.input = x;
        return .{
            .data = std.math.exp(x.data),
        };
    }

    pub fn backward(self: Self, gy: f32) DeZeroError!f32 {
        const x = self.input orelse return DeZeroError.BackwardBeforeForward;
        const gx = std.math.exp(x.data) * gy;
        return gx;
    }
};

pub fn numericalDiff(f: anytype, x: Variable, eps: f32) f32 {
    var op = f;
    const x0: Variable = .{ .data = x.data - eps };
    const x1: Variable = .{ .data = x.data + eps };
    const y0: Variable = op.forward(x0);
    const y1: Variable = op.forward(x1);

    return (y1.data - y0.data) / (2.0 * eps);
}

test "test Variable" {
    const x: Variable = .{
        .data = 1.0,
    };

    const eps: f32 = 1e-10;

    const expected: f32 = 1.0;
    const actual = x.data;
    try testing.expectApproxEqAbs(expected, actual, eps);
}

test "test Function" {
    const x: Variable = .{
        .data = 2.0,
    };

    const eps: f32 = 1e-10;
    var square: Square = .{};
    var exp: Exp = .{};

    var expected: f32 = 4.0;
    var actual: f32 = square.forward(x).data;
    try testing.expectApproxEqAbs(expected, actual, eps);

    expected = std.math.exp(2.0);
    actual = exp.forward(x).data;
    try testing.expectApproxEqAbs(expected, actual, eps);
}

test "test Function composition" {
    const v: Variable = .{
        .data = 2.0,
    };

    const eps: f32 = 1e-10;

    var op1 = Square{};
    var op2 = Exp{};
    var op3 = Square{};

    const expected: f32 = std.math.pow(f32, std.math.exp(std.math.pow(f32, 2.0, 2.0)), 2.0);
    const actual: f32 = op1.forward(op2.forward(op3.forward(v))).data;
    try testing.expectApproxEqAbs(expected, actual, eps);
}

test "test Numerical differentiation" {
    const x: Variable = .{
        .data = 2.0,
    };

    const eps: f32 = 1e-2;
    const square: Square = .{};

    const expected: f32 = numericalDiff(square, x, eps);
    const actual: f32 = 4.0;
    try testing.expectApproxEqAbs(expected, actual, eps);
}

test "test Backpropagation" {
    const eps: f32 = 1e-10;

    var op1 = Square{};
    var op2 = Exp{};
    var op3 = Square{};

    var x = Variable{ .data = 0.5 };
    var a = op1.forward(x);
    var b = op2.forward(a);
    var c = op3.forward(b);

    var expected: f32 = std.math.pow(f32, std.math.exp(std.math.pow(f32, 0.5, 2.0)), 2.0);
    var actual: f32 = c.data;
    try testing.expectApproxEqAbs(expected, actual, eps);

    c.grad = 1.0;
    b.grad = try op3.backward(c.grad.?);
    a.grad = try op2.backward(b.grad.?);
    x.grad = try op1.backward(a.grad.?);

    expected = 3.2974426293330694;
    actual = x.grad.?;
    try testing.expectApproxEqAbs(expected, actual, eps);
}
