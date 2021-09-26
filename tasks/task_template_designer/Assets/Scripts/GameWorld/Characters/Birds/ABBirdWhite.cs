using UnityEngine;
using System.Collections;

public class ABBirdWhite : ABBird {

	private Rigidbody2D _eggRigidBody;
	public float _eggForce    = 1f;
	public float _pushUpForce = 1f;

	void InitSpecialPower() {

		GameObject obj = (GameObject) Instantiate (ABWorldAssets.EGG, transform);
		obj.transform.parent = transform.parent;
		obj.name = "Egg";

		_eggRigidBody = obj.GetComponent<Rigidbody2D> ();
		_eggRigidBody.gameObject.SetActive(false);
	}

	void SpecialAttack() {

        ABGameWorld.Instance.KillBird(this);

        _eggRigidBody.transform.position = transform.position;
		_eggRigidBody.gameObject.SetActive (true);
		_eggRigidBody.AddForce (Vector2.down * _eggForce, ForceMode2D.Impulse);
		_rigidBody.AddForce(Vector2.up * _pushUpForce, ForceMode2D.Impulse);
	}
}
