# tf_transformations

## Context
In ROS 1, the TF library provided the helpful [`transformations.py`](https://github.com/ros/geometry/blob/noetic-devel/tf/src/tf/transformations.py) module for doing various rotation-based conversions.

Quoting from [the ROS wiki page](http://wiki.ros.org/geometry2/RotationMethods)
> The tf package also includes the popular transformations.py module. TransformerROS uses transformations.py to perform
> conversions between quaternions and matrices. transformations.py does has[sic] useful conversion on numpy matrices;
> it can convert between transformations as Euler angles, quaternions, and matrices.

Since TF was supported in all distros of ROS 1, the library was widely used for all sorts of angular conversions
in `rospy` code.

However, ROS 2 only supports [TF2](https://wiki.ros.org/tf2) (not to be confused with [TF2](https://store.steampowered.com/app/440/Team_Fortress_2/)) and TF2 does not have an equivalent library.

Quoting from TF (Tully Foote) himself on [ROS Answers](https://answers.ros.org/question/373068/ros2-foxy-tftransformationsquaternion_from_euler-equivalent/),
> tf.transformations is a fork of [https://github.com/cgohlke/transformations/](https://github.com/cgohlke/transformations/). This package has been deprecated "Transformations.py is no longer actively developed and has a few known issues and numerical instabilities."

The recommended alternative is a package called [`transforms3d`](https://matthew-brett.github.io/transforms3d/). This was originally only available via `pip` but is now available via `apt` and other package managers [in the `rosdistro`](https://github.com/ros/rosdistro/pull/33091).

However, using that library has a few obstacles that make porting ROS 1 code to ROS 2 difficult.
 1. The API is different. The new API has more consistent naming, but even then, it is not a one-to-one translation. for example, `tf.transformations.quaternion_from_euler` could be replaced with `transforms3d.euler.euler2quat`, but `tf` returns the quaternion with the ordering `x, y, z, w` and `transforms3d` returns `w, x, y, z`.
 2. Not all of the functions have been migrated. As noted in the [`transforms3d` Refactoring Plan](https://matthew-brett.github.io/transforms3d/devel/refactor_plan.html), some of the functions are still in the "to integrate" pile.

## Migration
If you're here, its likely because you want to migrate some code easily. You have two options:
 1. Use `transforms3d` by adding a dependency on `python3-transforms3d` in your `package.xml` and noting differences in API discussed above.
 2. Use this library `tf_transformations`.

If you wrote the following in ROS 1,

    from tf.transformations import euler_from_quaternion

the only change you need to your code in ROS 2 is

    from tf_transformations import euler_from_quaternion

Note that ROS 1 uses `tf` DOT `transformations` and ROS 2 uses `tf` UNDERSCORE `transformations`.

You also need to add a dependency on the `tf_transformations` in your `package.xml`

## Installation

This package is available in full as a ROS binary, which you can install via

    sudo apt install ros-$ROSDISTRO-tf-transformations

In older versions, you would need to manually install the `transforms3d` library via pip.

    sudo pip3 install transforms3d

## Implementation
Wherever possible, this library uses `transforms3d` to do the math for each of the functions. For functions that
are still on the Refactoring Plan, the original implementation is left in place. It is not ideal to have to maintain
the math for those functions, given the complexity of the geometry at hand, and in the future, effort should be made
to fix the problems upstream and replace the implementations in this package with the upstream versions.

However, the original `transformations.py` library was remarkably stable. There have only been [two commits](https://github.com/ros/geometry/commits/noetic-devel/tf/src/tf/transformations.py) in its entire Git history (the initial commit and porting to Python3, which was a one line cosmetic change). Therefore, the hope is that this package will not need too much maintenance to correct problems.

## Special Thanks
This package rests on the shoulders of giants, namely the OG author [Christoph Gohlke](http://www.lfd.uci.edu/~gohlke/) [[github]](https://github.com/cgohlke/transformations) and [Matthew Brett](https://matthew-brett.github.io/transforms3d/).
