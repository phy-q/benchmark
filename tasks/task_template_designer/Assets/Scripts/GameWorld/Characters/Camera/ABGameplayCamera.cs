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

﻿using UnityEngine;
using System.Collections;

public class ABGameplayCamera : MonoBehaviour {

	private Rect  _initialCameraRect;
	private Camera _camera;
	
	private bool  _isDraging;
	private float _dragDistance;

	public float _dampTime;
	public float _minWidth;
	public float _maxWidth;

	void Awake()
	{
		_dragDistance = 1f;
		_initialCameraRect = CalculateCameraRect();
		_minWidth = _initialCameraRect.width;

		_camera = GetComponent<Camera> ();
	}
	
	void Update()
	{
		// Updating camera zoom
		CalculateOrthographicSize();

		if(!_isDraging)
		{
			ABBird target = ABGameWorld.Instance.GetCurrentBird();

			Vector3 cameraNextPos = Vector3.zero;
			cameraNextPos.x = LeftBound();

			if(target && target.OutOfSlingShot && target.IsInFrontOfSlingshot())
			{
				cameraNextPos.x = RightBound()/2f;
			}
			else
			{
				if(_dragDistance < 0f)

					cameraNextPos.x = RightBound();
				else
					cameraNextPos.x = LeftBound();
			}

			FollowTarget(cameraNextPos);
		}

		_isDraging = false;
	}

	void FollowTarget(Vector3 targetPosition)
	{
		Vector3 cameraPos = transform.position;
		cameraPos.x = Mathf.Lerp(cameraPos.x, targetPosition.x, _dampTime * Time.deltaTime);
		transform.position = cameraPos;
	}

	public float LeftBound()
	{
		return _initialCameraRect.width/2f - _maxWidth/2f;
	}

	public float RightBound()
	{
		return _maxWidth/2f - _initialCameraRect.width/2f;
	}
	
	public void DragCamera(Vector3 dragDistance)
	{
		ABBird target = ABGameWorld.Instance.GetCurrentBird();

		if(target && target.IsFlying)
			return;

		_isDraging = true;

		if(dragDistance.magnitude > 0f)
			_dragDistance = dragDistance.x;

		Vector3 cameraPos = transform.position;
		cameraPos.x = Mathf.Lerp(cameraPos.x, cameraPos.x - dragDistance.x, _dampTime * Time.deltaTime);
		cameraPos.x = Mathf.Clamp(cameraPos.x, LeftBound(), RightBound());
		transform.position = cameraPos;
	}
	
	public void SetCameraWidth(float width)
	{
		_initialCameraRect.width = width;
		_initialCameraRect.width = Mathf.Clamp(_initialCameraRect.width,  _minWidth, _maxWidth);
	}
	
	public void ZoomCamera(float zoomFactor)
	{
		if(!ABGameWorld.Instance._isSimulation)
		{
			_initialCameraRect.width += zoomFactor;
			_initialCameraRect.width = Mathf.Clamp(_initialCameraRect.width,  _minWidth, _maxWidth);
		}
	}

	public Rect CalculateCameraRect()
	{
		float height = 2f * GetComponent<Camera>().orthographicSize;
		float width = height * GetComponent<Camera>().aspect;	

		return new Rect(transform.position.x - width/2f, transform.position.y - height/2f, width, height);
	}

	public float CalculateOrthographicSize()
	{
		float orthographicSize = GetComponent<Camera>().orthographicSize;
		
		Vector3 topRight = new Vector3(_initialCameraRect.x + _initialCameraRect.width, _initialCameraRect.y, 0f);
		Vector3 topRightAsViewport = GetComponent<Camera>().WorldToViewportPoint(topRight);
		
		if (topRightAsViewport.x >= topRightAsViewport.y)

			orthographicSize = Mathf.Abs(_initialCameraRect.width) / GetComponent<Camera>().aspect / 2f;
		else
			orthographicSize = Mathf.Abs(_initialCameraRect.height) / 2f;
		
		_camera.orthographicSize = Mathf.Lerp(GetComponent<Camera>().orthographicSize, orthographicSize, _dampTime * Time.deltaTime);
		
		return orthographicSize;
	}
}
