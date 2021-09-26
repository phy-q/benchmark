// SCIENCE BIRDS: A clone version of the Angry Birds game used for 
// research purposes
// 
// Copyright (C) 2016 - Lucas N. Ferreira - lucasnfe@gmail.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>
//

using UnityEngine;
using System.Collections;

public class ABSlingshot : MonoBehaviour {
	
    public Vector3 _positionOffset;
    public float _width;

    LineRenderer _lineRenderer;

    void Start() {

        _lineRenderer = gameObject.GetComponent<LineRenderer>();

        _lineRenderer.material = new Material(Shader.Find("Custom/Solid Color"));
        _lineRenderer.SetWidth(_width, _width);

		_lineRenderer.SetPosition((int)SLINGSHOT_LINE_POS.SLING, transform.position + _positionOffset);
        _lineRenderer.enabled = false;
    }

    void Update()
    {
		if(_lineRenderer && ABGameWorld.Instance.IsSlingshotBaseActive())
        {
            _lineRenderer.enabled = true;
			_lineRenderer.SetPosition((int)SLINGSHOT_LINE_POS.BIRD, ABGameWorld.Instance.GetSlingshotBasePosition());
        }
        else

            _lineRenderer.enabled = false;
    }
}
