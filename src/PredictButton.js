class PredictButton extends React.Component {
    render() {
        return (
            <button type="button"
                    id={this.props.id}
                    className="btn btn-primary w-50 border border-dark"
                    onClick={this.props.onClick}
                    style={{maxHWidth: '200px',
                        marginTop: '10px',
                        marginBottom: '15px '}}>
                Predict Points
            </button>
        )
    }
}