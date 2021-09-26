Shader "Custom/Solid Color" {

    Properties {
        _Color ("Main Color", Color) = (0.08, 0, 0, 1)
    }

    SubShader {
        Pass { Color [_Color] }
    }
}
